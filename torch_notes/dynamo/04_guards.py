from typing import List
import torch

compilation_count = 0

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global compilation_count
    compilation_count += 1
    print(f">>> Compilation #{compilation_count}")
    gm.graph.print_tabular()
    return gm

@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x

print("=== Call 1: x.shape=(10,), y.shape=(10,) ===")
foo(torch.randn(10), torch.ones(10))

print("\n=== Call 2: same shapes  guards PASS, no recompilation ===")
foo(torch.randn(10), torch.ones(10))

# --- Different shape: guards fail, recompilation ---
print("\n=== Call 3: x.shape=(20,), y.shape=(20,)  shape guard FAILS, recompile ===")
foo(torch.randn(20), torch.ones(20))

# --- Different dtype: guards fail, recompilation ---
print("\n=== Call 4: same shape but x.dtype=float64  dtype guard FAILS, recompile ===")
foo(torch.randn(10, dtype=torch.float64), torch.ones(10, dtype=torch.float64))

# --- Different device (if cuda available): guards fail ---
if torch.cuda.is_available():
    print("\n=== Call 5: same shape but on cuda  device guard FAILS, recompile ===")
    foo(torch.randn(10, device="cuda"), torch.ones(10, device="cuda"))
else:
    print("\n=== Call 5: skipped (no CUDA) ===")

print(f"\n{'='*60}")
print(f"Total compilations: {compilation_count} (one per unique guard combination)")
total_calls = 5 if torch.cuda.is_available() else 4
print(f"Total calls: {total_calls}")
print(f"Cache hits (guards passed): {total_calls - compilation_count}")
print(f"{'='*60}")


x = torch.randn(10)
y = torch.ones(10)
explanation = torch._dynamo.explain(foo, x, y)
print(f"\n=== torch._dynamo.explain() ===")
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Graphs compiled: {explanation.graph_count}")
print(f"\n  Guards ({len(explanation.out_guards)} total):")
for g in explanation.out_guards:
    if g.code_list:
        print(f"    [{g.source}] {g.guard_types}: {g.code_list}")
    else:
        print(f"    [{g.source}] {g.guard_types or g.name}")


# TORCH_LOGS=guards python3 dynamo/03_guards.py
# TORCH_LOGS=verbose_guards python3 dynamo/03_guards.py
# TORCH_LOGS=recompiles python3 dynamo/03_guards.py
