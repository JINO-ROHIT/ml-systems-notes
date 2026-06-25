import torch

@torch.compile
def foo(x):
    if x.sum() > 2:
        x += 1
    else:
        x -= 1
    return x

x = torch.tensor([5.0])
print(foo(x))


# TORCH_LOGS=graph_breaks python3 dynamo/05_graph_break.py
# TORCH_LOGS=graph_code python3 dynamo/05_graph_break.py
