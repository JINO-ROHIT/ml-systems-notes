"""
this is on a L4

modal run triton/modal_runner.py --script 08_layouts.py --args "R_vs_throughput"
R vs. Throughput
================
R=1   0.200 TB/s
R=2   0.210 TB/s
R=4   0.215 TB/s
R=8   0.217 TB/s
R=16  0.219 TB/s

modal run modal_runner.py --script 08_layouts.py --args "LDG_STG_instructions"

R=1
==========
--:-:2:-:1      @!P5 LDG.E R13, [R8.64+0x200];
--:-:3:-:1      @!P1 LDG.E R21, [R8.64+0xa00];
--:-:4:-:1      @!P4 LDG.E R15, [R8.64+0x400];
--:-:3:-:1      @!P0 LDG.E R23, [R8.64+0xc00];
--:-:3:-:1      @!P6 LDG.E R11, [R8.64];
--:-:3:-:1      @!P3 LDG.E R17, [R8.64+0x600];
--:-:3:-:1      @!P1 LDG.E R29, [R8.64+0x1400];
--:-:3:-:1      @!P0 LDG.E R25, [R8.64+0x1000];
--:-:3:-:1      @!P2 LDG.E R19, [R8.64+0x800];
--:0:3:-:1      @!P1 LDG.E R12, [R2.64];
--:-:3:-:4      @!P0 LDG.E R27, [R8.64+0x1200];
--:-:3:-:4      @!P4 LDG.E R35, [R8.64+0x1a00];
--:-:3:-:4      @!P2 LDG.E R31, [R8.64+0x1600];
--:-:3:-:4      @!P3 LDG.E R33, [R8.64+0x1800];
--:-:3:-:1      @!P5 LDG.E R37, [R8.64+0x1c00];
--:1:5:-:3      @!P6 LDG.E R36, [R4.64];
04:-:-:-:1      @!P2 STG.E [R6.64+0x200], R13;
16:0:-:-:1      @!P0 STG.E [R6.64+0x400], R15;
08:1:-:-:1      @!P3 STG.E [R6.64], R11;
--:1:-:-:1      @!P2 STG.E [R6.64+0x600], R17;
--:1:-:-:1      @!P0 STG.E [R6.64+0xa00], R21;
--:1:-:-:1      @!P3 STG.E [R6.64+0x800], R19;
--:1:-:-:1      @!P2 STG.E [R6.64+0xc00], R23;
--:1:-:-:1      @!P1 STG.E [R2.64], R12;
--:1:-:-:1      @!P0 STG.E [R6.64+0x1000], R25;
--:1:-:-:1      @!P4 STG.E [R6.64+0x1a00], R35;
--:1:-:-:4      @!P2 STG.E [R6.64+0x1600], R31;
--:1:-:-:4      @!P1 STG.E [R6.64+0x1400], R29;
--:1:-:-:4      @!P0 STG.E [R6.64+0x1200], R27;
--:1:-:-:4      @!P3 STG.E [R6.64+0x1800], R33;
--:1:-:-:1      @!P5 STG.E [R6.64+0x1c00], R37;
32:-:-:-:1      STG.E [R14.64], R36;

R=2
==========
--:0:2:-:1      @!P6 LDG.E.64 R8, [R2.64];
--:0:3:-:4      @!P5 LDG.E.64 R10, [R2.64+0x400];
--:0:4:-:4      @!P4 LDG.E.64 R12, [R2.64+0x800];
--:0:3:-:4      @!P3 LDG.E.64 R14, [R2.64+0xc00];
--:0:3:-:4      @!P2 LDG.E.64 R16, [R2.64+0x1000];
--:0:3:-:4      @!P1 LDG.E.64 R18, [R2.64+0x1400];
--:0:3:-:1      @!P0 LDG.E.64 R20, [R2.64+0x1800];
--:1:5:-:1      @!P6 LDG.E.64 R22, [R4.64];
04:1:-:-:1      @!P5 STG.E.64 [R6.64], R8;
16:1:-:-:4      @!P4 STG.E.64 [R6.64+0x800], R12;
08:1:-:-:6      @!P3 STG.E.64 [R6.64+0xc00], R14;
--:1:-:-:4      @!P5 STG.E.64 [R6.64+0x400], R10;
--:1:-:-:4      @!P2 STG.E.64 [R6.64+0x1000], R16;
--:1:-:-:4      @!P1 STG.E.64 [R6.64+0x1400], R18;
--:1:-:-:1      @!P0 STG.E.64 [R6.64+0x1800], R20;
32:-:-:-:1      STG.E.64 [R2.64], R22;

R=4
==========
--:-:2:-:4      @!P1 LDG.E.128 R12, [R2.64+0x800];
--:-:3:-:4      @!P0 LDG.E.128 R8, [R2.64];
--:-:4:-:1      @!P2 LDG.E.128 R16, [R2.64+0x1000];
--:0:5:-:4      @!P3 LDG.E.128 R20, [R2.64+0x1800];
04:0:-:-:4      @!P1 STG.E.128 [R4.64+0x800], R12;
08:0:-:-:4      @!P0 STG.E.128 [R4.64], R8;
16:0:-:-:1      @!P2 STG.E.128 [R4.64+0x1000], R16;
32:-:-:-:1      STG.E.128 [R4.64+0x1800], R20;

R=8
==========
--:-:2:-:1      @!P1 LDG.E.128 R16, [R2.64+0x1000];
--:0:5:-:5      @!P1 LDG.E.128 R20, [R2.64+0x1010];
--:0:5:-:4      @!P0 LDG.E.128 R8, [R2.64];
--:0:5:-:1      @!P0 LDG.E.128 R12, [R2.64+0x10];
04:0:-:-:1      @!P1 STG.E.128 [R4.64+0x1000], R16;
32:1:-:-:4      STG.E.128 [R4.64], R8;
--:1:-:-:2      STG.E.128 [R4.64+0x10], R12;
32:-:-:-:1      STG.E.128 [R4.64+0x1010], R20;

R=16
==========
--:-:2:-:1      @!P0 LDG.E.128 R16, [R2.64+0x20];
--:0:5:-:4      @!P0 LDG.E.128 R20, [R2.64+0x30];
--:0:5:-:4      @!P0 LDG.E.128 R8, [R2.64];
--:0:5:-:1      @!P0 LDG.E.128 R12, [R2.64+0x10];
04:0:-:-:1      @!P0 STG.E.128 [R4.64+0x20], R16;
32:1:-:-:4      STG.E.128 [R4.64], R8;
--:1:-:-:2      STG.E.128 [R4.64+0x10], R12;
32:-:-:-:1      STG.E.128 [R4.64+0x30], R20;
"""
import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")

@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # The main difference between writing this kernel in Triton and Gluon is
    # we need to specify the layout of the 1D tensor. Layouts are propagated
    # forwards through type inference, so we only need to specify the layout for
    # the indices tensor.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel


def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)


@pytest.mark.parametrize("XBLOCK", [128, 256])
@pytest.mark.parametrize("xnumel", [200, 1000])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_1d(XBLOCK, xnumel, num_warps):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1], [32], [num_warps], [0])
    memcpy_1d_impl(input, output, XBLOCK, layout, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


def main():
    run_throughput = _enabled("R_vs_throughput")
    show_instructions = _enabled("LDG_STG_instructions")
    if not (run_throughput or show_instructions):
        return

    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
    compiled_kernels = []

    if run_throughput:
        print("R vs. Throughput")
        print("================")
    else:
        # One tile is enough to compile and inspect each layout without benchmarking.
        input = torch.zeros(XBLOCK, device="cuda")
        output = torch.empty_like(input)

    for i in range(5):
        R = 2**i
        layout = gl.BlockedLayout(
            size_per_thread=[R],
            threads_per_warp=[32],
            warps_per_cta=[num_warps],
            order=[0],
        )
        impl = partial(kernel, layout=layout)
        if run_throughput:
            compiled_kernel, throughput = bench_memcpy(impl)
            print(f"R={R:<3} {throughput:.3f} TB/s")
        else:
            compiled_kernel = impl(input, output)
        compiled_kernels.append((R, compiled_kernel))

    if show_instructions:
        print("\nLDG/STG instructions")
        print("====================")
        for R, compiled_kernel in compiled_kernels:
            print(f"\nR={R}")
            print("==========")
            sass = compiled_kernel.asm["sass"]
            for line in sass.splitlines():
                if "LDG.E" in line or "STG.E" in line:
                    print(line)
    print()


if __name__ == "__main__":
    main()
