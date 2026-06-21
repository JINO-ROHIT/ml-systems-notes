"""
not supported but dynamo can support this
"""

import torch

def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:

        return torch.neg(x)

traced = torch.fx.symbolic_trace(func_to_trace)