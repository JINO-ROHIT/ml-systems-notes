"""
for every row we spawn a program and then process the row in chunks of BLOCK_SIZE
"""

import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def row_sum_kernel(
    x_ptr, out_ptr,
    N, stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    total = tl.full((), 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            x_ptr + row * stride_row + cols * stride_col,
            mask=cols < N,
            other=0.0,
        ).to(tl.float32)

        total += tl.sum(values, axis=0)

    tl.store(out_ptr + row, total)


def row_sum(x):
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=torch.float32)

    row_sum_kernel[(M,)]( # for every row we get a program
        x, output,
        N, x.stride(0), x.stride(1),
        BLOCK_SIZE=128,
    )

    return output



def main():
    torch.manual_seed(0)
    A = torch.randn((10, 200), device="cuda")
    output_torch = A.sum(dim=1, dtype=torch.float32)
    output_triton = row_sum(A)
    torch.testing.assert_close(output_triton, output_torch)
    print(f"PASS; max difference="
            f"{(output_torch - output_triton).abs().max().item()}")


if __name__ == "__main__":
    main()
