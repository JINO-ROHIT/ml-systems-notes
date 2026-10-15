import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # 1d launch grid
    curr_block = pid * BLOCK_SIZE
    offsets = curr_block + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

    output = x + y

    tl.store(out_ptr + offsets, output, mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    torch.manual_seed(0)
    # Include partial blocks to exercise the load/store mask.
    for size in (1, 1023, 1024, 1025, 98432):
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        output_torch = x + y
        output_triton = add(x, y)
        torch.testing.assert_close(output_triton, output_torch)
        print(f"size={size}: PASS; max difference="
              f"{(output_torch - output_triton).abs().max().item()}")


if __name__ == "__main__":
    main()
