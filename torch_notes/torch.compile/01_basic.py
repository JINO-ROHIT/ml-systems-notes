import torch

torch._logging.set_logs(graph_code=True) # to turn on logging

t1 = torch.randn(10, 10)
t2 = torch.randn(10, 10)

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

print(opt_foo2(t1, t2))