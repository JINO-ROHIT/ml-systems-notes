"""
1. max = max(row)
2. num = exp(element - max)
3. denom = sum(num)
4. softmax = num/denom

in this kernel, we do a a block size = whole row
"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    curr_block = pid * BLOCK_SIZE
    offsets = curr_block + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n

    row = tl.load(x_ptr + offsets, mask)
    
    max = tl.max(row)
    num = tl.exp(row - max)
    denom = tl.sum(num)

    softmax = num / denom

    tl.store(out_ptr + offsets, softmax, mask)



def softmax(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),) 
    softmax_kernel[grid](x, output, n_elements, BLOCK_SIZE = 1024)
    return output

def main():
    torch.manual_seed(0)
    size = 1024 # should match block size
    num_rows = 10
    x = torch.rand(num_rows, size, device = DEVICE)
    output_torch = F.softmax(x, dim = 1)
    output_triton = softmax(x)
    torch.testing.assert_close(output_triton, output_torch)
    print(f"size={size}: PASS; max difference="
            f"{(output_torch - output_triton).abs().max().item()}")


if __name__ == "__main__":
    main()