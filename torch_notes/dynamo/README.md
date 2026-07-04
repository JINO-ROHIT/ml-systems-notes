## torch dynamo

dynamo is a tracer that given and function and inputs to it, it executes the function and records a linear sequence of instructions (without control flow) into a graph.

from `01.py`, you can see that foo() captures a computation graph from the function, and dynamo saves the captured computation graph as an fx graph.

### python bytecode

dynamo captures the computation graph during the translation of Python bytecode. before execution, python functions are compiled into bytecode by the python virtual machine. each instance of a function corresponds to something called a frame, bascially stores the global variables, local variables, bytecode, and other information required to run the function.

the python virtual machine is a stack-based machine, which maintains three stacks:

call stack - tracks which functions are currently being called. When `foo()` calls `bar()` calls `baz()`, the call stack holds three frames: `baz` (top), `bar`, `foo` (bottom). Each frame stores that function's local variables, bytecode, and execution state. When a function returns, its frame is popped off.

evaluation stack (or data stack) - each frame has its own eval stack where bytecode instructions push/pop intermediate values. For example, `a + b` compiles to: `LOAD_FAST a` (pushes `a`), `LOAD_FAST b` (pushes `b`), `BINARY_OP +` (pops two values, pushes result). 

block stack - each frame has a block stack that tracks nested control structures like loops, `try`/`except`, `with` blocks. When you enter a `for` loop, a block marker is pushed; when you exit, it's popped. This tells the VM what to do when it sees `break` or `continue` - those operations target the innermost loop block on the stack. The block stack is purely about control flow (loops, exception handling), not data.

torch dynamos compilation process occurs before execution ie it's a JIT compiler. when Python is about to execute a function, torchDynamo begins translating the bytecode and capturing the computation graph. it does this by using a Python virtual machine simulator that constructs a corresponding computation graph while simulating the execution of Python bytecode.

### guards

guards are a very important concept.

after you capture the computation graph, you need to verify if you can still use the same graph for the next executions.
TorchDynamo creates a Python executable function called guard from the compiled function. it is responsible for checking whether the input attributes of the compiled function have changed. if they haven't changed, the previously compiled function can be reused; otherwise, the current input is invalid for the previously compiled function, and the function needs to be recompiled . 

all the functions compiled by dynamo are stored in a per-function cache to avoid recompilation for repeated inputs. by default `cache_size_limit = 8`, meaning at most 8 unique guard/input combinations are cached. once this limit is exceeded, dynamo logs a warning and falls back to eager mode i think.

also not every input change triggers a new compilation. dynamo can make shape dimensions dynamic, so two different shapes may share a compiled graph if dynamo decides the dimension varies. 

but guards on dtype, device, or other tensor properties always cause recompilation because those cannot be dynamic.


### graph breaks

dynamo cannot capture all functions onto a single computation graph. when TorchDynamo encounters an operator it cannot support, it creates a graph break, splitting the computation graph into several subgraphs that it can support, and returns the result to the Python interpreter to execute the operator it could not handle.

this is usually where the slow down happens and you need to be very careful to handle this scenario.


### statis vs dynamic shapes

by default, TorchDynamo uses static shape mode, where the tensor's shape and stride are recorded when capturing the computation graph. at the end of the computation graph capture , the guards generated check if the input information in the computation graph has changed . if no changes have occurred, the already compiled computation graph is reused; otherwise, the computation graph is recaptured and recompiled (graph recompilation). from the second time, the static shape is now captured in dynamic shape mode.

If you know that a dimension will vary in size, you can mark it as dynamic by calling torch._dynamo.mark_dynamic before calling torch.compile. This will avoid the first compilation with a static shape



my personal blog version - https://jino-rohit.github.io/blogs/13_dynamo.html