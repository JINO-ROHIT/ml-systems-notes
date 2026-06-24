### how to use torch compile

ref - https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

torch compile is a method to speed up pytorch code after 2.0 using JIT compilation and requires almost litle to no change.
any python fn or pytorch module can be passed and will be replaced by the optimized one.

torch.compile takes extra time to compile the model on the first few executions. torch.compile re-uses compiled code whever possible, so if we run our optimized model several more times, we should see a significant improvement compared to eager. check ex 03_speedup.py
(improve thsi definition later on)


#### the compilation stack

![](../artifacts/compilation_stack.png)

this is the overall flow for the torch compile stack from my understanding - 

1. pytorch api - this is your regular nn.module that you write in torch.
2. dynamo - dynamo intercepts the regular python flow and captures these pytorch specific operations into a graph. you can think of them like DAGS.
3. fx graph - fx graph is pytorch's internal graph representation. this IR is pretty easy to work with and debug since its just graphs and it has only 6 main instructions.
4. aten ops - all the operation captured in the graph have to be lowered to the primitives written in C++ in torch, for instance cos, sin etc. all of them are present in the aten/ library.
5. torch inductor - this is the actual compiler backend that takes these aten ops, and finally lowers them into triton kernels and ptx and so on.



#### graph breaks

The graph break is one of the most fundamental concepts within torch.compile. It allows torch.compile to handle arbitrary Python code by interrupting compilation, running the unsupported code, then resuming compilation. The term “graph break” comes from the fact that torch.compile attempts to capture and optimize the PyTorch operation graph. When unsupported Python code is encountered, then this graph must be “broken”. Graph breaks result in lost optimization opportunities, which may still be undesirable, but this is better than silent incorrectness or a hard crash.
