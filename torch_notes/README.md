### a guide to reading pytorch source code.

as ive been using torchtitan a lot more, sometimes the kernels that are invoked arent what i expected. i realized i needed to learn to profile and have atleast an intermediate understanding of how to trace an op end to end. (also this is a major signal if you can understand this very well)

pytorch is a massive library so cloning it and looking at code line by line probably doesnt work. this is my guide to the anatomy of torch and how to start reading it.


pytorch mainly has a four-layer architecture -
- python frontend - this is the python facing API where you write things like nn.Module, torch.tensor when you are building your favorite llm.
- dispatcher - when you call one matmul, it has to decide whether to use the CPU, CUDA, or MPS for computation. the dispatcher does this for you.
- C++ backend (ATen / c10) - i think this is where all the mathematical operations and memory management happens.
- compiler stack - this is a more recent feature that came with 2.0. torch dynamo captures the computation graph while the inductor optimizes and generates code.


always remember these four main layers.

1. the python layer

start with torch/nn/modules/module.py - the base class for all models along with the hooks are defined here.
you could also pickj an op of your choice and trace the __call__ method.
for the c++ binding side, torch/csrc/ has the pybind11 code that converts python objects to c++ pointers.


2. the dispatcher

torch.matmul(a, b) doesnt directly jump to a handwritten kernel.

there is a map called the native_functions.yaml at aten/src/ATen/native/native_functions.yaml. it lists every operator, its dispatch keys, and which c++ function implements it. 


for instance look at this, the grouped_mm disptaches to _scaled_grouped_mm_cuda.

```
- func: _scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
  variants: function
  dispatch:
    CUDA: _scaled_grouped_mm_cuda
  tags: needs_exact_strides
```

but if you want to see the real c++, you need to compile a debug build of pytorch.


3. ATen

this is where all the math operators and functions need to be define. source is in aten/src/ATen/native/.

there is a nice README.md guide within the folder on how to add this.


4. compiler stack

torch.compile is literally free performance when you use it well.

torch dynamo reads the python bytecode and captures the computation graph. inductor compiles that graph into triton kernels.

to debug the graphs. use this instead:

  TORCH_LOGS="+dynamo,+inductor" python your_model.py

the code is under torch/_dynamo/ and torch/_inductor/. its all python but its too dense. i still havent figured out whats the best way to start reading this section.



more resources

- pytorch advanced section is quite nice
- pytorch developer podcast hosted by edward yang. i wished they continued this but i think its stopped now
- ezyang's blog on pytorch internals

you will also need to build a debug version of torch and keep the source code generated during the compilation process otherwise, it will be difficult to find the source of some functions in the function call stack.

you can try this -

1. pick the main branch for instance and then -

```
export DEBUG=1
python setup.py bdist_wheel
uv pip install dist/torch*.whl
```

2. you can launch a torch script you want to debug and launch using gdb, start adding breakpoints and watch the whole function stack.



the best thing to do would be to trace just a single sufficiently complex operation end to end and not try to read the entire codebase and self destruct. ill try to add more details to this as i discover more.

happy reading!


further reading
1. fx graph paper - https://arxiv.org/pdf/2112.08429
2. fx graph documentation - https://docs.pytorch.org/docs/2.12/fx.html
3. torch compile manual - https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.ivdr7fmrbeab
4. https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
5. https://docs.pytorch.org/functorch/nightly/notebooks/aot_autograd_optimizations.html
6. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html#dynamo-deep-dive
7. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html
8. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_faq.html
9. https://docs.pytorch.org/docs/2.12/user_guide/torch_compiler/torch.compiler_faq.html

