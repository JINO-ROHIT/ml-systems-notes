import torch
torch._logging.set_logs(graph_code=True)
torch._logging.set_logs(graph_breaks=True) # to see the graph breaks

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = torch.compile(bar)
inp1 = torch.ones(10)
inp2 = torch.ones(10)

torch._dynamo.reset() # reset to clear the torch.compile cache
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)


"""
when you call bar the first time, we see two graphs being traced, for the torch abs part + the b < 0 part
in the second time, the torch abs part is cached, so only b < 0 part runs
"""


"""In order to maximize speedup, graph breaks should be limited. We can force TorchDynamo to raise an error upon the first graph break encountered 
by using fullgraph=True"""

"""
When TD encounters unsupported Python syntax, such as data-related control flow, it exits the computation graph, 
allowing the Python interpreter to handle the unsupported code, and then continues capturing the graph. 
Specifically: Before encountering the conditional branch `if b.sum() < 0`, TD captures the graph and executes normally. 
Upon encountering the conditional branch, TD lets Python determine the branch's outcome. 
"""
import traceback as tb
torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
try:
    opt_bar_fullgraph(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()