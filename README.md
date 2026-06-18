# ml-systems-notes

a personal collection of notes on ml systems engineering covering distributed computing, parallelism, quantization, and pytorch internals.

> everything here is a work in progress. i add notes as i do experiments and projects.


## contents

- [distributed-techniques](./distributed-techniques/) - distributed training fundamentals: nccl collectives (gather, all-gather, reduce, all-reduce, scatter, reduce-scatter), mixture-of-experts, parallelism strategies (dp, ddp, zero, tensor/pipeline parallelism), and torch.distributed basics.

- [quantization](./quantization/) - model quantization from first principles: symmetric/asymmetric quantization, llm.int8(), awq, smoothquant, gptq/obs/obq, and quip.

- [torch-notes](./torch-notes/) - pytorch internals

- [jax-scaling-book](./jax-scaling-book/) - roofline analysis exercises for matrix multiplication in jax/tpu context.

