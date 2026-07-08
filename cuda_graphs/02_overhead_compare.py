import torch
import time

def tiny_kernels(n_kernels: int = 200):
    """many tiny element-wise ops."""
    x = torch.randn(128, device="cuda")
    y = torch.randn(128, device="cuda")

    # --- eager ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_kernels):
        z = x + y
    torch.cuda.synchronize()
    t_eager = time.perf_counter() - t0

    # --- captured graph ---
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        z = x + y

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_kernels):
        g.replay()
    torch.cuda.synchronize()
    t_graph = time.perf_counter() - t0

    print(f"{n_kernels} launches of a tiny kernel")
    print(f"  eager: {t_eager*1e3:.2f} ms")
    print(f"  graph: {t_graph*1e3:.2f} ms")
    print(f"  speedup: {t_eager/t_graph:.1f}x")

if __name__ == "__main__":
    tiny_kernels(500)
