"""Four small Gluon kernels to study before 10_async_copy.py.

    modal run modal_runner.py --script 09_layout_practice.py
    modal run modal_runner.py --script 09_layout_practice.py --args "--step bias --show-layouts"
"""

import argparse

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def indices_kernel(out, M, N, L: gl.constexpr, R: gl.constexpr, C: gl.constexpr):
    # A 2D parent needs two 1D slices for its coordinate vectors.
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, L))
    # Expanding each vector restores L; broadcasting creates a 2D tensor.
    values = 100 * rows[:, None] + cols[None, :]
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    gl.store(out + offsets, values, mask)


@gluon.jit
def copy_kernel(x, out, M, N, L: gl.constexpr, R: gl.constexpr, C: gl.constexpr):
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, L))
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    # The pointer tensor, loaded values, and store all use parent layout L.
    values = gl.load(x + offsets, mask, other=0)
    gl.store(out + offsets, values, mask)


@gluon.jit
def row_bias_kernel(x, bias, out, M, N, L: gl.constexpr, R: gl.constexpr, C: gl.constexpr):
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, L))
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = gl.load(x + offsets, mask, other=0)
    # One bias per row, with the same sliced layout as rows.
    row_bias = gl.load(bias + rows, rows < M, other=0) # picks the first 5 entry
    # [R] -> [R, 1] restores L, then broadcasts across columns.
    result = values + row_bias[:, None] # changes to column
    gl.store(out + offsets, result, mask)


@gluon.jit
def row_sum_kernel(x, out, M, N, L: gl.constexpr, R: gl.constexpr, C: gl.constexpr):
    rows = gl.arange(0, R, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, L))
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    # Zero padding matters: invalid columns must contribute zero to the sum.
    values = gl.load(x + offsets, mask, other=0)
    # Removing columns by reduction produces SliceLayout(1, L).
    # This matches rows, so each result can be stored using out + rows.
    totals = gl.sum(values, axis=1)
    gl.store(out + rows, totals, rows < M)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("all", *LESSONS), default="all")
    parser.add_argument("--layout", choices=("compact", "kernel"), default="compact")
    parser.add_argument("--show-layouts", action="store_true")
    args = parser.parse_args()

    if args.layout == "compact":
        layout = gl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    else:
        layout = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    # One program handles the whole small matrix. Padding exercises the masks.
    R, C = 16, 8
    M, N = 5, 7
    x = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    bias = 100 * torch.arange(M, device="cuda", dtype=torch.float32)
    print(f"Parent layout: {layout}")
    print(f"Logical input: {M}x{N}; padded tile: {R}x{C}; one program, four warps.")
    print("Changing the layout changes thread ownership, not the expected values.")
    print("Input:\n", x.cpu())
    if args.show_layouts:
        for name, item, shape in (
            ("Parent", layout, [R, C]),
            ("Rows: SliceLayout(1, parent)", gl.SliceLayout(1, layout), [R]),
            ("Columns: SliceLayout(0, parent)", gl.SliceLayout(0, layout), [C]),
        ):
            print(f"\n{name}\n{item.format_tensor_view(shape)}")

    steps = LESSONS if args.step == "all" else [args.step]
    for step in steps:
        print(f"\n{step.upper()}: {LESSONS[step]}")
        if step == "sum":
            out = torch.empty(M, device="cuda", dtype=x.dtype)
            row_sum_kernel[(1,)](x, out, M, N, layout, R, C, num_warps=4)
            expected = x.sum(dim=1)
        else:
            out = torch.empty_like(x)
            if step == "indices":
                indices_kernel[(1,)](out, M, N, layout, R, C, num_warps=4)
                expected = bias[:, None] + torch.arange(N, device="cuda")[None, :]
            elif step == "copy":
                copy_kernel[(1,)](x, out, M, N, layout, R, C, num_warps=4)
                expected = x
            else:
                print("Row bias:", bias.cpu())
                row_bias_kernel[(1,)](x, bias, out, M, N, layout, R, C, num_warps=4)
                expected = x + bias[:, None]
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        print(out.cpu())
        print("PASS: matches PyTorch exactly")


if __name__ == "__main__":
    main()
