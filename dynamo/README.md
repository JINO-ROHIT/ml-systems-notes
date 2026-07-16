### personal notes on nvidia dynamo

dynamo is a kubernetes native serving framework that makes serving across multiple gpus efficient and optimal. it has many important features -

1. disagg serving - splits the prefill and decode phases across different gpus.
2. kv aware routing - routes the user request to the appropriate node where the kv cache is present.
3. low latency communication library (NIXL) -  low latency point-to-point inference data transfer library that accelerates the transfer of KV cache between GPUs and across heterogeneous memory and storage types.
4. kv block manager - a cost aware kv caching engine that transfers KV cache across various memory hierarchies, freeing up GPU memory while maintaining user experience.

detailed examples - https://github.com/ai-dynamo/dynamo/blob/v0.8.1/examples/
