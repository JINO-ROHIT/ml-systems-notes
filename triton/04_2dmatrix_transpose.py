import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(a_ptr, b_ptr, stride_am, stride_an, stride_bm, stride_bn, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    curr_row_blk = row * BLOCK_SIZE_M
    curr_col_blk = col * BLOCK_SIZE_N

    curr_row_offs = curr_row_blk + tl.arange(0, BLOCK_SIZE_M) 
    curr_col_offs = curr_col_blk + tl.arange(0, BLOCK_SIZE_N) 

    a_ptr = a_ptr + curr_row_offs[:, None] * stride_am + curr_col_offs[None, :] * stride_an
    b_ptr = b_ptr + curr_row_offs[:, None] * stride_bn + curr_col_offs[None, :] * stride_bm

    a_mask = (curr_row_offs[:, None] < M) & (curr_col_offs[None, :] < N)
    b_mask = (curr_row_offs[:, None] < M) & (curr_col_offs[None, :] < N)

    a = tl.load(a_ptr, a_mask)

    tl.store(b_ptr, a, b_mask)
    

def matrix_transpose(a, b):
    M, N = a.shape
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    transpose_kernel[grid](a, b, a.stride(0), a.stride(1), b.stride(0), b.stride(1), M, N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32)
    return b


def main():
    A = torch.randn((10, 200), device="cuda")
    B = torch.randn((200, 10), device="cuda")

    B_result = matrix_transpose(A, B)
    print(torch.equal(A, B_result.T))


if __name__ == "__main__":
    main()