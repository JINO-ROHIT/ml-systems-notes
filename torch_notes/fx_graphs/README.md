### what are fx graphs?

torch.fx is a tool that captures programs via symbolic tracing, represents them using a simple 6-instruction python-based IR, and re-generates python code from the IR to execute it.

what is symbolic tracing? 

this is the way in which torch.fx converts python code into a graph without actually computing real values.
it feeds fake values, called proxies, through the code and the operations on these proxies are recorded.


the intermediate representation is the operations that were recorded during symbolic tracing. it consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or torch.nn.Module instances), and return values.

```
symbolic tracing -> intermediate representation -> transforms -> Python code generation
```


an example for the graph created by torch.fx

```
    %x : [num_users=1] = placeholder[target=x]
    %linear_weight : [num_users=1] = get_attr[target=linear.weight]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %relu : [num_users=1] = call_method[target=relu](args = (%linear,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu,), kwargs = {dim: -1})
    %topk : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
    return topk
```

these are the 6 IR operations every fx graph can have - 

| op              | meaning                      |
| --------------- | ---------------------------- |
| `placeholder`   | graph input                  |
| `get_attr`      | read module parameter/buffer |
| `call_function` | call a Python function       |
| `call_method`   | call a method on an object   |
| `call_module`   | call a submodule             |
| `output`        | graph output                 |


1. the main limitation of symbolic tracing is it does not currently support dynamic control flow. if loops or if statements where the condition may depend on the input values of the program.

```
def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:

        return torch.neg(x)
```

2. non torch functions also are not supported but you can wrap them to record them in the trace.

```
import torch
import torch.fx
from math import sqrt

def normalize(x):
    return x / sqrt(len(x))

# It's valid Python code
normalize(torch.rand(3, 4))

traced = torch.fx.symbolic_trace(normalize) # wont work


torch.fx.wrap('len')
torch.fx.wrap('sqrt')

traced = torch.fx.symbolic_trace(normalize) # now works

```

3. static control flow is supported where loops or if statements whose value cannot change across invocations.


```
class MyModule(torch.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        # This if-statement is so-called static control flow.
        # Its condition does not depend on any input values
        if self.do_activation:
            x = torch.relu(x)
        return x
```