import torch

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

# Test that it works
a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]
ref = fn(a, b, c, d)
loss = ref.sum()
loss.backward()

from functorch.compile import aot_function


def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

# Pass on the compiler_fn to the aot_function API
aot_print_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

# Run the aot_print_fn once to trigger the compilation and print the graphs
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs
res = aot_print_fn(cloned_a, cloned_b, cloned_c, cloned_d)
res.sum().backward()
assert torch.allclose(ref, res)

"""
the forward graph

def forward(self, primals_1, primals_2, primals_3, primals_4):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, primals_3);  add = primals_3 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, primals_4);  add_1 = primals_4 = None
    cos = torch.ops.aten.cos.default(add_2)
    cos_1 = torch.ops.aten.cos.default(cos)
    return (cos_1, add_2, cos)

the backward graph

def forward(self, add_2, cos, tangents_1):
    sin = torch.ops.aten.sin.default(cos);  cos = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
    sin_1 = torch.ops.aten.sin.default(add_2);  add_2 = None
    neg_1 = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg_1);  mul = neg_1 = None
    return (mul_1, mul_1, mul_1, mul_1)
"""



# using the min cut partition

from functorch.compile import min_cut_rematerialization_partition

# Zero out the gradients so we can do a comparison later
a.grad, b.grad, c.grad, d.grad = (None,) * 4

# Lets set up the partitioner. Also set the fwd and bwd compilers to the printer function that we used earlier.
# This will show us how the recomputation has modified the graph.
aot_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=min_cut_rematerialization_partition)
res = aot_fn(a, b, c, d).sum().backward()


"""
def forward(self, primals_1, primals_2, primals_3, primals_4):
    add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, primals_3);  add = primals_3 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, primals_4);  add_1 = primals_4 = None
    cos = torch.ops.aten.cos.default(add_2)
    cos_1 = torch.ops.aten.cos.default(cos);  cos = None
    return (cos_1, add_2)
    


def forward(self, add_2, tangents_1):
    cos = torch.ops.aten.cos.default(add_2)
    sin = torch.ops.aten.sin.default(cos);  cos = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
    sin_1 = torch.ops.aten.sin.default(add_2);  add_2 = None
    neg_1 = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg_1);  mul = neg_1 = None
    return (mul_1, mul_1, mul_1, mul_1)

    
we can see that compared to default partitioner, forward pass now outputs fewer tensors, and recomputes some operations in the backward pass
"""