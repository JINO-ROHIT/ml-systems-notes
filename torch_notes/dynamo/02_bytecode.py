import torch

@torch.compile
def foo(x, y):
    return (x + y) * x

x = torch.randn(10)
y = torch.ones(10)
foo(x, y)


# TORCH_LOGS=bytecode python3 dynamo/02_bytecode.py 