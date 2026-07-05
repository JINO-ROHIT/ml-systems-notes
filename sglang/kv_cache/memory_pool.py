from __future__ import annotations

from typing import List, Optional

import torch


class Req:
    """A dummy stripped down request"""

    def __init__(self, rid: str):
        self.rid = rid
        # assigned by ReqToTokenPool.alloc(); None until then.
        self.req_pool_idx: Optional[int] = None

        # used in the alloc() assertion where a request that already has
        # committed KV (from a prior chunk) or is in the middle of a
        # chunked prefill is allowed to reuse its existing slot.
        self.kv_committed_len: int = 0
        self.inflight_middle_chunks: int = 0

    def __repr__(self):
        return f"Req(rid={self.rid}, pool_idx={self.req_pool_idx})"

class ReqToTokenPool:
    """maps each request to its token locations in the KV cache.

    the pool is a 2D int32 tensor:
        shape = (size + 1, max_context_len)
    """

    def __init__(self, size: int, max_context_len: int, device: str = "cuda"):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device

        # +1 for the padding row at index 0
        self._alloc_size = size + 1

        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len),
            dtype=torch.int32,
            device=device,
        )

        # free slot but row 0 is never used
        self.free_slots: List[int] = list(range(1, self._alloc_size))

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        """assign a pool row to each new request in the batch.

        requests that already have `req_pool_idx` (e.g. continuing a
        chunked prefill) keep their existing slot.  Returns the list of
        assigned indices, or None if there are not enough free slots.
        """
        # Requests that will reuse their existing slot
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]

        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need = len(reqs) - len(reusing)
        if need > len(self.free_slots):
            return None

        selected = self.free_slots[:need]
        self.free_slots = self.free_slots[need:]

        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = selected[offset]
                offset += 1

        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req) -> None:
        """Return a request's pool row to the free list."""
        assert req.req_pool_idx is not None, "double free?"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def read(self, req: Req) -> torch.Tensor:
        """Read the token-slot row for a single request."""
        assert req.req_pool_idx is not None
        return self.req_to_token[req.req_pool_idx]

    def available_size(self) -> int:
        return len(self.free_slots)

    def clear(self) -> None:
        """Reset the entire pool."""
        self.free_slots = list(range(1, self._alloc_size))
        self.req_to_token.zero_()

    def __repr__(self):
        return (
            f"ReqToTokenPool(size={self.size}, "
            f"max_context_len={self.max_context_len}, "
            f"free={len(self.free_slots)}, "
            f"device={self.device})"
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = ReqToTokenPool(size=4, max_context_len=8, device=device) # maximum 4 requests in the pool

    print("pool created")
    print(pool)
    print(f"req_to_token shape: {pool.req_to_token.shape}")
    print()

    # allocate
    reqs = [Req("req_a"), Req("req_b"), Req("req_c")]
    indices = pool.alloc(reqs)
    print(f"Allocated 3 reqs in pool indices: {indices}")
    print(f"Available slots: {pool.available_size()}")
    for r in reqs:
        print(f"  {r}")
    print()

    # write token locations
    # Suppose req_a has 3 tokens at slots [10, 11, 12]
    pool.write(
        torch.tensor([reqs[0].req_pool_idx], device=device),
        torch.tensor([[10, 11, 12, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device),
    )
    print(f"after write, req_a row: {pool.read(reqs[0]).tolist()}")
    print()

    # allocate more than available should return None
    more = [Req("req_d"), Req("req_e")]
    result = pool.alloc(more)
    print(f"alloc 2 more but only 1 free slot left {result}")
    print()

    # free one 
    pool.free(reqs[1])
    print(f"freed req_b. available: {pool.available_size()}")
    print()

    # allow now
    result = pool.alloc([Req("req_f")])
    print(f"alloc req_f {result}")
    print()

    pool.clear()
    print(f"after clear, available: {pool.available_size()}")
    print(f"req_to_token is zeroed: {pool.req_to_token.sum().item() == 0}")
