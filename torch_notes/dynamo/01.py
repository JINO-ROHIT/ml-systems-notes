from typing import List
import torch

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm

@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x

if __name__ == "__main__":
    a, b = torch.randn(10), torch.ones(10)
    foo(a, b)


### another way to do this is to use torch logs


import torch

@torch.compile
def foo(x, y):
    return (x + y) * x

x = torch.randn(10)
y = torch.ones(10)
foo(x, y)


# TORCH_LOGS=graph_code python3 dynamo/01.py 