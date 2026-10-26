# ref - https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html

import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    # For the 1D indices, use a SliceLayout along the dimensions we will expand.
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # expand_dims along the slice dimension returns a tensor with the parent
    # layout, so this yields [XBLOCK, 1] and [1, YBLOCK] tensors with the same
    # layout which can be broadcasted together to [XBLOCK, YBLOCK].
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # Compute the mask the same way: select for indices along each dimension
    # that are in bounds and broadcast them together.
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    # Pass the strides of the input and output tensors into the kernel. The
    # compiler will specialize the kernel if any of the strides are 1, which is
    # common for the inner dimension of tensors.
    compiled_kernel = memcpy_2d_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return compiled_kernel


@pytest.mark.parametrize("XBLOCK, YBLOCK", [(128, 256), (256, 128)])
@pytest.mark.parametrize("xnumel, ynumel", [(100, 2000), (1000, 200)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_2d(XBLOCK, YBLOCK, xnumel, ynumel, transposed, num_warps):
    torch.manual_seed(0)
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    # Transposing the tensor makes it non-contiguous along the inner dimension.
    input = input.T if transposed else input
    output = output.T if transposed else output
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    memcpy_2d_impl(input, output, XBLOCK, YBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)

def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)

def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)

def bench_memcpy_2d(impl, transposed=False):
    # 8 GB tensor, but spread across 2 dimensions.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    input = input.T if transposed else input
    output = output.T if transposed else output
    return bench_memcpy_impl(input, output, impl)

def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")

if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    print("Benchmarking 2D memcpy")
    print("======================")
    XBLOCK = 1
    YBLOCK = 2048
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=XBLOCK, YBLOCK=YBLOCK, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl)
    print(f"Throughput: {throughput:.3f} TB/s")

# what if we transpose
if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Transposed throughput: {throughput:.3f} TB/s")

# Because the inner dimension is no longer contiguous, we get no coalescing. Simply swapping the block sizes and transposing the layout restores performanc
if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_2d(impl, transposed=True)
    print(f"Fixed throughput: {throughput:.3f} TB/s")
    print()


if __name__ == "__main__" and _enabled("memcpy_2d_contig"):
    print("Non-contiguous memcpy")
    print("=====================")
    # 8 GB tensor.
    xnumel = 32 * 1024
    ynumel = 64 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    # Take a view over every other row.
    input = input[::2]
    output = torch.empty_like(input)
    assert not input.is_contiguous() and output.is_contiguous()

    # Benchmark 2D memcpy.
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy: {throughput:.3f} TB/s")

    # Benchmark PyTorch contiguous.
    fn = lambda: input.contiguous()
    ms = triton.testing.do_bench(fn)
    throughput = get_throughput(input, ms)
    print(f"torch.Tensor.contiguous: {throughput:.3f} TB/s")

    # We can eke out even more performance by using the transposed "trick".
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input.T, output.T, impl)
    print(f"2D memcpy (transposed): {throughput:.3f} TB/s")
    print()

if __name__ == "__main__" and _enabled("memcpy_2d_inout"):
    print("2D memcpy in/out layouts")
    print("=========================")

    # Input is contiguous along dim 1.
    input = torch.randn((32 * 1024, 32 * 1024), device="cuda")

    # Output is contiguous along dim 0.
    output = torch.empty((input.shape[1], input.shape[0]), device="cuda").T

    # order=[1, 0]
    layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    impl = partial(memcpy_2d_impl, XBLOCK=1, YBLOCK=2048, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[1, 0]): {throughput:.3f} TB/s")

    # order=[0, 1]
    layout = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    impl = partial(memcpy_2d_impl, XBLOCK=2048, YBLOCK=1, layout=layout, num_warps=4)
    _, throughput = bench_memcpy_impl(input, output, impl)
    print(f"2D memcpy (order=[0, 1]): {throughput:.3f} TB/s")