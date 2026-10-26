# good explanation for swizzle layouts - https://github.com/triton-lang/triton/discussions/2026

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


def is_ampere_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


if __name__ == "__main__" and not is_ampere_or_newer():
    raise RuntimeError("This tutorial requires Ampere or newer NVIDIA GPU")

@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)

    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    # For 1D tensor, pick a simple layout.
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    # Issue the async copy.
    cp.async_load(smem, in_ptr + offsets, mask=mask)
    # `commit_group` puts all previously issued async copies into a group.
    cp.commit_group()

    # Wait until the number of pending groups reaches 0. Then we can retrieve
    # the data from shared memory.
    cp.wait_group(0)

    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)


def memcpy_1d_cpasync(input, output, XBLOCK=8192, num_warps=4):
    grid = (triton.cdiv(input.numel(), XBLOCK), )
    memcpy_1d_cpasync_kernel[grid](input, output, input.numel(), XBLOCK, num_warps=num_warps)


@pytest.mark.parametrize("xnumel, XBLOCK", [(200, 128), (1000, 256)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_memcpy_1d_cpasync(xnumel, XBLOCK):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_cpasync(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


@gluon.jit
def elementwise_add_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)

    # Compute the offset to the row this program will process.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        # Offset to the column block.
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
def test_elementwise_add(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


@gluon.jit
def elementwise_add_cpasync_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    # New: declare shared memory for the A tile and B tile.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # Issue loads for both A and B tiles.
        cp.async_load(a_smem, a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        cp.async_load(b_smem, b_ptrs + ystride_b * yoffs[None, :], mask=mask)
        # Commit both loads to the same group.
        cp.commit_group()
        # Wait until both loads are complete!
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add_cpasync(A, B, C, smem_layout, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_cpasync_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires Ampere or newer")
def test_elementwise_add_cpasync(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    elementwise_add_cpasync(a, b, c, smem_layout, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


def get_throughput(ms, C):
    # Because this kernel is memory-bound, we will measure bandwidth.
    tbytes = (3 * C.numel() * C.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    ms = triton.testing.do_bench(lambda: elementwise_add(A, B, C))
    print(f"elementwise_add: {get_throughput(ms, C):.2f} TB/s")

    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    ms = triton.testing.do_bench(lambda: elementwise_add_cpasync(A, B, C, smem_layout))
    print(f"elementwise_add_cpasync: {get_throughput(ms, C):.2f} TB/s")
