## MOE (mixture of experts)

moes are a type of sparse model where unlike the transformers which have FFN(feed forward layers), it replaces them by experts.
these experts are simply the actual FFN layer but broken down into smaller pieces.

the core idea is given a question, you route the question to multiple experts, and then combine the response from each expert to give the final result.

so what do we need to achieve this?

1. a router - the intelligent layer that takes in the question and decides which expert to use.
2. the actual expert itself which are ffn layers.

now training this is the tricky part. 
- say we have 10 experts, when the weights are intiialized to zeros, each of them are equally likely to be chosen. we need to be able to choose only the appropriate experts.
- also if certain experts are more preferred over the others, they will always end up being chosen.  


solution
1. use a top-k method to select experts - say instead of having all the 10 experts active at all time, we choose only top-k experts to be active. this ensures we dont use all the experts. the router takes care of this.
2. another technique is to add some noise so we occasionally mix up and the training becomes more stable over time.


cool, now we have a basic idea of moe's in general. it time to solidify our understanding and reason about this architecture when an input comes in.

im taking the example of qwen3-30B since ive been exclusively working with it. it has 128 expert layers but only 8 of them are active.

say we have an input sentence of 100 words. the router routes 10 words to expert 1, and 50 words to expert 2, and so on.

1. balancing load - most of the tokens are being routed to a certain expert, which cause load on that expert while the other experts are chilling untrained. we need to find a nice mechanism to balance the load.
2. when you want to scale these experts across multiple gpus, we use a technique called expert parallelism, where each layer is distributed across N gpus nodes.in this case, if the only one expert is being loaded, the other gpus are just unterutilized and boooo, lower mfu.

the gshard paper proposes -
1. expert capacity - what if we enforce a limit to how many tokens a certain expert can process? this way load is equal. the overflowing tokens activations dont change and are simply carried over as connections to the next layer.
2. a similar auxiliary loss like noise to balance out training blah blah.

the problem with this approach is simply finding the right capacity value. if its too high, you end up loosing a lot of tokens and this affects training.