import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import symbolic_trace


def f(x, y):
    return x + y

x = torch.randn(8)
y = torch.randn(8)

print("make_fx")
g = make_fx(f)(x, y)
print(g.code)

"""
def forward(self, x_1, y_1):
    add = torch.ops.aten.add.Tensor(x_1, y_1);  x_1 = y_1 = None
    return add
"""

print("symbolic_trace")
h = symbolic_trace(f)
print(h.code)


"""
def forward(self, x, y):
    add = x + y;  x = y = None
    return add
"""

# key difference:
#   make_fx -> torch.ops.aten.add.Tensor   (low-level, ATen IR)
#   symbolic_trace -> x + y                (high-level, Python ops)

######################################################################################################################

# the duplicate tensor problem 
# make_fx maps tensors to FX nodes by tensor ID (id()).
# passing the same tensor twice means both uses map to the SAME proxy node.

def f(x, y):
    return x + y

x = torch.randn(8)

print("=== 2. make_fx with duplicate tensor ===")
g = make_fx(f)(x, x)           # same tensor passed as both x and y
print(g.code)
# expected: x + x   actual: y + y 

# why? the tracer sees two args with the same id and maps them to one proxy.
# the second arg (y) reuses the same proxy, and the later one wins the name.
# this means the traced graph is semantically wrong.
