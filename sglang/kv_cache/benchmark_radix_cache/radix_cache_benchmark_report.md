# SGLang Radix Cache Benchmark Qwen3.5-0.8B

i ran a experiment to understand the effect of radix cache in sglang. it wasnt what i had excepted but radix cache turned off seemed to be perform better than with radix cache enabled!

i benchmarked Qwen/Qwen3.5-0.8B (hybrid: Gated DeltaNet + Attention) on an A100 for 50 requests with 10 warmups across varying sequence lengths and types of workload. the workload was completed different tokens in a sequence, identical sequence and shared sequences(system prompt shared) across short, medium and long sequences. 


across almost every scenario i tested, turning radix cache off was faster. the only case where radix cache turned on pulled ahead was with identical 2000 token prompts, where it was 9% faster.


my guess is for such a small model, the actual overhead of constructing the tree, splitting nodes and looking up prefixes is more than just computing the actual kv cache tensors itself.


there is also a small gotcha in pip version 0.5.9 where enabling radix cache on hybrid Mamba models silently forces disable_overlap_schedule=True. this is a separate, much larger performance hit unrelated to the cache itself. i used --mamba-scheduler-strategy extra_buffer to avoid this.
