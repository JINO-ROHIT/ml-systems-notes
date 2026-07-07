## cuda graphs

whats the point of having cuda graph?

there are two things that needs to happen every time you execute a CUDA kernel -
1. the actual computation itself, say matrix multiplication computes on some matrix and stores the result.
2. the cpu has to launch this kernel, it needs to do a set of preliminary tasks for instance preparing parameters, sending request to the driver, and the driver
queuing the request to run on the gpu.

in most cases where computation is heavy, the cpu side overhead is mostly negligible, it doesnt really affect us all that much.
in cases where computation isnt happening at all that much, a very short kernel, then the cpu side overhead starts to be noticable.

say you have 100 kernels, but they are very short. in this case the cpu overhead takes longer than the actual computation itself to run on the gpu.
the idea of cuda graph is instead of launching all of these kernels over and over again, why dont we remember this specific sequence of kernels to run, 
package it into a nice graph, and then just replay this graph each time.

a formal definition is cuda graphs describes a series of gpu jobs (kernel, memcpy, memset, host jobs) as a DAG, where each node represents an operation and edges represent dependencies. this DAG is instantiated into an executable object(GraphExec) which can then be launched repeatedly.

the biggest gain from using cuda graph is the cpu overhead being eliminated by a single lightweight graph launch. remember this does not make your computation itself faster.

can i always use cuda graphs as default then?

no, cuda graphs are useful only at specific places and you need to be thoughtful about how you use them.

1. creating the graph and the first launch is usually quite slow, depending on the actual operators that you have. a rough estimate is that the first launch is ~33% slower than the new few launches. because of this cost, applying it for a one time application doesnt make sense at all. the work needs to be highly repetitive for cuda graphs to make a difference.
2. if you have a lot of short kernels, this is probably a good place to apply them.
3. because it is a DAG which is a fixed structure, all the nodes and dependenecies, shapes and sizes need to be fixed during recording time. you can change them after. 


now stage do you think is it more applicable to use cuda graphs? prefill or decode?

decode! 

1. if you remember in prefill, you can get varying sequence length of the input from different users. but for decode, the output length token is always 1.
in both cases the batch sizes keep changing, so the inference engine usually captures multiple versions of the graph at different batch sizes. at runtime,  it simply pads to the nearest batch size and generates tokens. 
2. the computations itself are quite heavy in prefill compared to decode, so there isnt a way to guarantee if adding cuda graphs is going to help much.



what if we wanted to support for prefill?

### piecewise cuda graph

instead of capturing the entire forward pass as one graph, PCG splits the computation graph into pieces at something called "split points" (e.g. MoE dispatch ops, attention). these are the operations which need a dynamic shape. each piece is then captured as a separate cuda graph for a set of pre-defined token lengths (eg. 4, 8, 16, 32, 48). at runtime input is padded to the nearest captured size and each piece is replayed.

the operation that could not be captured run as eager.

1. torch.compile traces the model into a graph - when the first forward pass runs under PCG mode, torch.compile (dynamo) traces the entire model forward into an fx graph. if you remember fx graphs, all the operations are captured as nodes.

2. the graph gets split at uncapturable ops - the inference engine receives this fx graph, walks through every node looking for ops that cannot be cuda graph capotured. for example -
- attention depends on cu_seqlens and max_seqlen, which vary per batch even at the same total token count
- all-reduce
- moe dispatch
each split op gets its own single node submodule. all the other ops stays in the capturable submodules between them. The result is a stitching graph (split_gm) like this:

submod_0 (capturable) -> submod_1 (split: all-reduce) -> submod_2 (capturable) -> submod_3 (split: attention) -> 

3. for the ops captured, cuda graphs are recorded, is padded to the nearest captured size and each piece is replayed during runtime.
4. the operation that could not be captured run in eager mode without any cuda graph.

this eliminates all the cpu overhead for most of the kernels like gemm, norm, rope etc and results in a speedup.

