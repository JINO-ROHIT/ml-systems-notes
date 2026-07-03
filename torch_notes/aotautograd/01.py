import torch
from functorch.compile import aot_function, \
    make_boxed_func

def fn(a, b):
    return a * b

def run_func(func, *inputs):
    res = func(*inputs)
    loss = res.sum()
    loss.backward()

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return make_boxed_func(fx_module.forward)

a, b = [torch.randn(2, 4, requires_grad=True,
    device="cuda") for _ in range(2)]
run_func(fn, a, b)

aot_print_fn = aot_function(fn, fw_compiler=compiler_fn,
    bw_compiler=compiler_fn)
run_func(aot_print_fn, a, b)


"""
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_2)
    return (mul, primals_1, primals_2)

think of this as -
primals = original inputs so primals_1 = a, primals_2 = b
mul = a * b
Returns (mul, a, b) basically the output plus saved tensors needed for backward


def forward(self, primals_1, primals_2, tangents_1):
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  primals_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = primals_2 = None
    return (mul_2, mul_1)


think of this as -
tangents_1 = incoming gradient (dloss/doutput)
Since y = a * b:
- dy/da = b  grad_a = tangents_1 * b (line: mul_2)
- dy/db = a  grad_b = tangents_1 * a (line: mul_1)
Returns (grad_a, grad_b)
"""