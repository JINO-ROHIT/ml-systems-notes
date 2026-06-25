import torch
import torch._dynamo as dynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woooooo")
    if b.sum() < 0:
        b = b * -1
    return x * b

explanation = dynamo.explain(
    toy_example, torch.randn(10), torch.randn(10))

# print(explanation)
print(f"\n=== torch._dynamo.explain() ===")
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Graphs compiled: {explanation.graph_count}")