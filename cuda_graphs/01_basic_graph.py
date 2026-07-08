import torch

def basic_graph_demo():
    N = 1024
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")

    # warmup
    for _ in range(10):
        c = a @ b

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        c = a @ b

    # replay (same inputs)
    for _ in range(5):
        g.replay()
    print(c)

    # replay with new inputs (same shape)
    a2 = torch.randn(N, N, device="cuda")
    b2 = torch.randn(N, N, device="cuda")

    # need to copy into the captured memory
    a.copy_(a2)
    b.copy_(b2)
    g.replay()
    print(c)

if __name__ == "__main__":
    basic_graph_demo()
