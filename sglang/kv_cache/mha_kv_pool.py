"""
MHATokenToKVPool

holds the physical K/V cache tensors on GPU. each attention layer gets
one K buffer and one V buffer. the attention backend reads from these
buffers during the forward pass and writes to them via set_kv_buffer().
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024

class RadixAttention:
    """minimal stub"""
    def __init__(self, layer_id: int):
        self.layer_id = layer_id

    def __repr__(self):
        return f"RadixAttention(layer_id={self.layer_id})"


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    ``loc`` is the full-pool write location; ``swa_loc`` is the pre-translated
    full->SWA location for hybrid SWA pools (``None`` otherwise). Bundling them
    lets a backend issue one ``set_kv_buffer`` call regardless of pool type.
    """
    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc)`` from a ``KVWriteLoc`` or a bare loc tensor."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc
    return loc_info, None

def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,                
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    """
    # original has 4 paths: JIT store_cache kernel, CPU AMX, cuda-graph
    # alt-stream overlap, and this naive fallback.  We only keep the fallback.
    """
    k_cache[indices] = k
    v_cache[indices] = v


class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """
        stripped down kv cache
        """
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        # FP8 storage workaround: torch can't index_put_ into fp8 tensors,
        # so we store as uint8 and view-cast on access.
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.mem_usage = 0

    def get_kv_size_bytes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()



class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size, page_size, dtype, layer_num, device,
            start_layer, end_layer,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        # Create the physical K/V buffers per layer
        self._create_buffers()
        self.device_module = torch.get_device_module(self.device)

        # For the store-cache write path
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

        self._log_allocation(size)

    def _log_allocation(self, num_tokens: int):
        k_size, v_size = self.get_kv_size_bytes()
        k_gb, v_gb = k_size / GB, v_size / GB
        logger.info(
            f"KV Cache allocated. dtype: {self.dtype}, #tokens: {num_tokens}, "
            f"K: {k_gb:.2f} GB, V: {v_gb:.2f} GB"
        )
        self.mem_usage = k_gb + v_gb

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _create_buffers(self):
        # Shape per layer: [size + page_size, head_num, head_dim]
        # The +page_size padding slot (index 0) is same trick as ReqToTokenPool row 0).
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.v_head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64, device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64, device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.shape[1] * x.shape[2] * x.dtype.itemsize for x in self.k_buffer + self.v_buffer],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer


    def get_kv_size_bytes(self):
        k_bytes = sum(x.nbytes for x in self.k_buffer)
        v_bytes = sum(x.nbytes for x in self.v_buffer)
        return k_bytes, v_bytes

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)


    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)

        #  which layer we're writing to
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id

        # dequantize if the incoming tensor is in a different dtype
        #  example, the attention backend produced fp8 K/V but pool stores bf16).
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k / k_scale
            if v_scale is not None:
                cache_v = cache_v / v_scale
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        # view cast if dtype different
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        # write into the pool
        _set_kv_buffer_impl(
            cache_k, cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            size_limit=self.size + self.page_size,
            alt_stream=None,
            same_kv_dim=self.same_kv_dim,
        )

    def __repr__(self):
        k_bytes, v_bytes = self.get_kv_size_bytes()
        return (
            f"MHATokenToKVPool("
            f"size={self.size}, page_size={self.page_size}, "
            f"head_num={self.head_num}, head_dim={self.head_dim}, "
            f"v_head_dim={self.v_head_dim}, layers={self.layer_num}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"K={k_bytes // (1024**2)}MB, V={v_bytes // (1024**2)}MB)"
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a small pool of 16 token slots, 4 layers, 8 heads, dim=128
    pool = MHATokenToKVPool(
        size=16,
        page_size=1,
        dtype=torch.float16,
        head_num=8,
        head_dim=128,
        layer_num=4,
        device=device,
    )
    print(pool)
    print()

    for layer in range(4):
        k, v = pool.get_kv_buffer(layer)
        print(f"Layer {layer}: K {list(k.shape)}, V {list(v.shape)}")

    print()

    # wite K/V at slot 2 in all layers
    layer = RadixAttention(layer_id=0)
    loc = torch.tensor([2], device=device)

    cache_k = torch.randn(1, 8, 128, dtype=torch.float16, device=device)
    cache_v = torch.randn(1, 8, 128, dtype=torch.float16, device=device)

    pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    # read back and verify
    k_buf, v_buf = pool.get_kv_buffer(0)
    print(torch.allclose(k_buf[2], cache_k[0]))
    print(torch.allclose(v_buf[2], cache_v[0]))
