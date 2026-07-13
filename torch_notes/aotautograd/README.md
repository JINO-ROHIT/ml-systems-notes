### understanding AOTautograd in pytorch

before the whole 2.0 ecosystem existed, users could capture the forward graph using torchfx tracing.
but although, each operator had its own forward and backward implementation,  users could not directly optimize on the backward graph, nor could they merge the forward and backward propagation computation graphs into a single computation graph.

AOT solves exactly this, making some optimizations possible for training.

with AOTAutograd, users can do the following -
1. obtain the backprop computation graph, or even the joint computation graph of forward and backprop
2. for training, you can perform joint optimization of forward and backward propagation. for example, reduce the number of tensors retained by forward propagation for backward propagation by recompiling in backward propagation, thereby reducing memory requirements.


so why is it called aotautograd?

the backpropagation computation graph in torch is dynamically constructed during the forward propagation process, while the backpropagation computation graph is only finalized at the end of the forward prop. AOTAutograd traces both forward and backpropagation simultaneously in an Ahead-of-Time manner obtaining the computation graphs for both propagation before the function is actually executed.

in general, the workflow of AOTAutograd is as follows -
1. the AOT dispatch traces forward and backward propagation and a joint forward and backward computation graph is generated, which is an FX graph containing the Aten/Prim operator.
2. The joint graph is divided into partition_fn, a forward propagation computation graph and a backward propagation computation graph.
3. Optional: By decompositions decomposing and sinking high-level operators down to smaller granular operators;
4. the forward propagation computation graph and the backward propagation computation graph are compiled separately and then integrated into one torch.autograd.Function


#### torch dispatch

pytorch has a dispatcher that acts like a router. every time you call an operator like `a * b`, the dispatcher decides which kernel to run based on the input tensors' properties. if the tensor is on CUDA, run the CUDA kernel, if it needs gradients, wrap it with autograd etc. an operator usually passes through multiple dispatch layers before reaching the final kernel.

__torch_dispatch__ is a hook that fires before the final kernel executes. it gives you access to the raw ATen operator and its inputs so you can intercept, inspect, or modify behavior at the op level.

one way to invoke is to use the torchdispatchmode.

torchdispatchmode is a context manager that intercepts every op called inside `with mode`, regardless of tensor type.

torchfx implements `make_fx`, which, unlike regular symbolic tracing, is implemented through __torch_dispatch__.

```
from torch.fx.experimental.proxy_tensor import make_fx

def f(x, y):
    return x + y

x = torch.randn(8)
y = torch.randn(8)
g = make_fx(f)(x, y)
print(g.code)
# def forward(self, x_1, y_1):
#     add = torch.ops.aten.add.Tensor(x_1, y_1)
```

`make_fx` maps tensors to FX graph nodes by tensor ID. this causes a problem when the same tensor is passed as multiple arguments because they map to the same proxy, producing an incorrect graph:

before tracing, AOTAutograd deduplicates inputs using two strategies:

1. detach repeated tensors into fresh leaf tensors. fast but breaks if the function applies in-place ops on the repeated tensor.
2. remove duplicate parameters from the function signature. produces a specialized graph for the duplicate case.

see `02_make_fx.py` for an example.


#### joint graph

wouldnt it be nice to have a single fx graph to cover both forward and backward, so you can optimize them (could be fusing ops etc).

1. build a join function.

```
def joint_forward_backward(*inputs):
    outputs = forward_fn(*inputs)            # run forward
    grads = torch.autograd.grad(
        outputs, inputs, grad_outputs=...
    )  # run backward
    return outputs, grads
```

it makes a wrapper than calls forward and then backward.

2. trace using make_fx

using the  __torch_dispatch__, AOTAutograd can trace the joint forward and backward propagation computation graph. if the user wants to optimize the forward prop function, AOTAutograd constructs and traces a joint_forward_backward function, which calls the forward propagation function which  then calls torch.autograd.grad to execute backward prop.

AOTAutograd make_fx traces this joint_forward_backward function, which triggers for each operator __torch_dispatch__. It retrieves a proxy from the tensor, fx.Graph creates the corresponding proxy for the operator call_function, sets the target to the operator itself, runs the operator with the actual tensor, and binds the resulting tensor to the proxy .

3. this process is repeated until AOTAutograd traces all operators in both forward and backward propagation, resulting in a complete joint graph.


#### partition

AOTAutograd partition_fn divides the joint graph into a seperate forward prop and backward prop computation graph. it has two built-in partition_fn types -

1. `default_partition` : same as pytorch's default behavior where it finds all operator outputs from the input to the output of the forward operation. Tensors used in the backward operation are also included as outputs of the forward operation, representing tensors that the forward operation reserves for the backward operation. All intermediate results of the forward are saved for the backward.

2. `min_cut_rematerialization_partition` : by introducing recomputation in the backward pass, it reduces the number of tensors retained in the forward pass, thus saving GPU memory. if you remember activation checkpointing, it kinda is similar to this. besides the input tensors in the backward pass  which must be retained, there are multiple options for retaining or discarding other tensors. however, determining how to select the tensors retained in the forward pass to achieve a tradeoff between computation and GPU memory is addressed here by solving a max-flow/min-cut problem. 

The process is as follows:
Add an edge between the source node (virtually added) and the primals (forward input Tensors), and add an edge between each tangent's closure (backward input Tensors) and the target node (sink, virtually added). These form a directed graph from the source to the sink, and the weights on the edges are the tensor size, which represents the amount of GPU memory used.
We need to find a suitable partitioning method to divide this directed graph into two parts such that the sum of the weights of the edges between the source subgraph and the target subgraph is minimized. This is a minimum cut problem.


(we need to look at the min cut theorem)

#### control statements

1. data dependent control flow is not supported in tracing
2. regular loops are unrolled during tracing
3. function calls are also expanded during tracing