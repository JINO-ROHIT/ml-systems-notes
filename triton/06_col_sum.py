"""
for every col we spawn a program and then process the cols in chunks of BLOCK_SIZE
"""

import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def col_sum_kernel(
    x_ptr, out_ptr,
    M, N, stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    col = tl.program_id(0) # we get the start of every column

    total = tl.full((), 0.0, tl.float32)
    for start in range(0, M, BLOCK_SIZE):
        rows = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            x_ptr + rows * stride_row + col * stride_col,
            mask=rows < M,
            other=0.0,
        ).to(tl.float32)

        total += tl.sum(values, axis=0)

    tl.store(out_ptr + col, total)


def col_sum(x):
    M, N = x.shape
    output = torch.empty(N, device=x.device, dtype=torch.float32)

    col_sum_kernel[(N,)]( # for every col we get a program
        x, output,
        M, N,  x.stride(0), x.stride(1),
        BLOCK_SIZE=128,
    )

    return output



def main():
    torch.manual_seed(0)
    A = torch.randn((10, 200), device="cuda")
    output_torch = A.sum(dim=0, dtype=torch.float32)
    output_triton = col_sum(A)
    torch.testing.assert_close(output_triton, output_torch)
    print(f"PASS; max difference="
            f"{(output_torch - output_triton).abs().max().item()}")


if __name__ == "__main__":
    main()
