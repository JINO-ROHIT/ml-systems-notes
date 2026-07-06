"""
sglang's automatic prefix caching for KV reuse.

when two requests share a common prefix ("what is the capital of"
+ "france?" / "italy?"), the kv values for the shared tokens are identical.
the radix tree detects this overlap and lets the second request reuse
the first request's kv slots instead of allocating new ones.

architecture:
    radixkey      - token sequence used as a tree key
    treenode      - a node in the radix tree, holding key + KV slot indices
    radixcache    - the tree itself where you can match_prefix / insert / evict
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import sys
import time
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class RadixKey:
    """a token sequence used as a radix tree key.

    is_bigram=True creates a bigram view over the raw tokens:
    if token_ids = [a, b, c], the logical key length is 2 and iteration
    yields (a,b), (b,c).

    extra_key adds a namespace tag like an ID so that
    requests with different extra_keys never share prefix nodes.
    """

    __slots__ = ("token_ids", "extra_key", "is_bigram")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        self.token_ids = token_ids
        self.extra_key = extra_key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        if self.is_bigram:
            n = len(self.token_ids)
            return n - 1 if n > 0 else 0
        return len(self.token_ids)

    def __iter__(self) -> Iterator:
        if self.is_bigram:
            t = self.token_ids
            for i in range(len(t) - 1):
                yield (t[i], t[i + 1])
        else:
            yield from self.token_ids

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        """slice the key, returning a new RadixKey sharing the underlying array."""
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        extra = f", extra_key={self.extra_key!r}" if self.extra_key else ""
        return (
            f"RadixKey({extra} "
            f"token_ids={list(preview)}{'...' if len(self.token_ids) > 10 else ''}, "
            f"is_bigram={self.is_bigram})"
        )

    def page_aligned(self, page_size: int) -> "RadixKey":
        """truncate to the largest multiple of page_size <= current length."""
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self, is_eagle: bool, value: Optional[torch.Tensor] = None
    ) -> Tuple["RadixKey", Optional[torch.Tensor]]:
        """Flip to bigram view in O(1) if *is_eagle* is true."""
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: "RadixKey") -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """Length of the shared prefix between *self* and *other*.

        Uses exponential-search + binary-search to avoid Python-level
        per-token looping on long shared prefixes.
        """
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        n = min(len(t0), len(t1))

        # Gallop in doubling windows, then binary-search the first divergence
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first page_size logical units,
        namespaced by *extra_key*."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)


