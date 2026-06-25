import torch

# default behavior
comp1 = [0]

def compiler1(gm, example_inputs):
    comp1[0] += 1
    print(f">>> Compilation #{comp1[0]}")
    gm.graph.print_tabular()
    return gm

@torch.compile(backend=compiler1)
def foo(x):
    return x * 2

print("=== Default: static to dynamic ===")
print("Call 1: shape (10,)  specialized, guard has size=[10]")
foo(torch.randn(10))

print("\nCall 2: shape (10,)  guard passes, cache hit")
foo(torch.randn(10))

print("\nCall 3: shape (20,)  guard fails, recompile. Now dynamic size=[None]")
foo(torch.randn(20))

print("\nCall 4: shape (5,)  guard passes (dynamic), no recompile")
foo(torch.randn(5))

print(f"\nTotal: {comp1[0]} compilations for 4 calls")


# what if we use explicit mark_dynamic
comp2 = [0]

def compiler2(gm, example_inputs):
    comp2[0] += 1
    print(f">>> Compilation #{comp2[0]}")
    return gm

@torch.compile(backend=compiler2)
def bar(x):
    return x * 2

print("\n\n=== Explicit mark_dynamic(x, 0) ===")
print("Call 1: shape (10,) but marked dynamic  no specialization")
x1 = torch.randn(10)
torch._dynamo.mark_dynamic(x1, 0)
bar(x1)

print("\nCall 2: shape (20,)  no recompile, already dynamic")
bar(torch.randn(20))

print("\nCall 3: shape (5,)  no recompile")
bar(torch.randn(5))

print(f"\nTotal: {comp2[0]} compilations for 3 calls")


print(f"\n{'='*60}")
print(f"Default:      {comp1[0]} compilations for 4 calls (recompiled once, then dynamic)")
print(f"mark_dynamic: {comp2[0]} compilation  for 3 calls (dynamic from start)")
print(f"{'='*60}")
print("Static: compiler knows exact size so faster kernels (unrolling, prefetching)")
print("Dynamic: one kernel fits all sizes so no recompilation, but less optimized")


#python3 dynamo/06_static_dynamic_shape.py
