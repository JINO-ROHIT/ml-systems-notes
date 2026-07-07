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