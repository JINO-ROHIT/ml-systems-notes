### personal notes on nvidia dynamo

dynamo is an open source, kubernetes native serving framework from nvidia for running llm inference across multiple gpus / nodes.

it sits above inference engines like tensorrt-llm, vllm, and sglang.

important features -

1. disagg serving - splits the prefill and decode phases across different gpus.
2. kv aware routing - routes the request to a worker that already has useful kv cache, while also considering load.
3. low latency communication library (nixl) - point-to-point inference data transfer, especially kv cache transfer between gpus and memory tiers.
4. kv block manager - manages kv blocks across gpu, cpu, ssd, and remote storage.
5. sla planner - automatically scales prefill / decode workers to hit ttft and itl targets.
6. dynamic gpu scheduling - adjusts replicas and placement based on demand.
7. fault tolerance - request migration, cancellation, graceful shutdown, request rejection.

detailed examples - https://github.com/ai-dynamo/dynamo/tree/main/examples/


### 1. pd disaggregation


1. why pd disaggregation works?

- prefill phase is compute intensive because it processes the entire input sequence.
    this is basically taking a (n, d) multiplied with another (d, n) = (n x n) matrix.
- decode phase is memory intensive because it generates one token at a time and reads the kv cache.
    here its more like (1, d) x (d, n) = (1, n).

so prefill is quite slow compared to decode and they have very different characteristics.

in a serving engine, what would you prioritize first? prefill reqs or decode reqs?

1. if you prioritize prefill, the ongoing chat / decode request waits behind a long prefill.
2. if you prioritize decode, new requests wait for their first token.

chunked prefill tries to fix this by capping how much prefill work can run at once.

- large chunk size = high throughput but decode waits longer.
- small chunk size = lower latency but lower gpu utilization.

piggyback prefill mixes prefill and decode in the same batch.

- decode has poor gpu utilization.
- mixing prefill with decode can use the gpu better.
- but colocated prefill/decode still has to make one compromise config.

pd disagg is the way to improve both axes because prefill and decode can be tuned separately.


2. how should you partition prefill and decode?

main questions -

1. how many prefill gpus?
2. how many decode gpus?
3. what parallelism strategy should each pool use?

tensor parallelism -

- splits a big tensor across multiple gpus.
- each gpu does less compute for the same operation.
- this makes each individual step faster, so it helps latency.
- decode is latency sensitive because users are waiting token by token.
- decode usually benefits more from tensor parallelism.

pipeline parallelism -

- splits model stages across gpus.
- while one request moves to the next gpu, the earlier gpu can start another request.
- this improves throughput because multiple requests can be in flight.
- but it adds communication between stages, so latency can go up.
- prefill is more throughput oriented, so pipeline parallelism fits prefill better.


3. batch size choice

- decode wants smaller batches.
    small batches produce the next token quickly.
- prefill wants larger batches.
    large batches keep gpu utilization high.

example numbers -

1. prefill batch can be something like 65k tokens.
2. decode batch can be much smaller, like 16 tokens.
3. prefill often has a first-token-latency budget around 1s to 10s.
4. decode often cares about per-token latency, roughly around 0.1s but it varies a lot.


4. rate matching

rate matching means matching prefill throughput with decode capacity.

- if there are too many prefill workers and too few decode workers, prefill produces kv cache faster than decode can consume it.
- then the queue between prefill and decode becomes the bottleneck.
- if there are too many decode workers, decode capacity sits idle while waiting for prefill.

the usual design process -

1. choose a first token latency target, like 5 seconds.
2. size the prefill pool so it can process the input context within that target.
3. size the decode pool so it can consume the kv cache and generate tokens at the same rate.
4. minimize total gpu count while meeting the sla.

dynamo has an sla planner for this.


5. when is pd disagg most useful?

pd disagg is best for prefill-heavy workloads.

prefill-heavy means -

- long input sequence length.
- shorter output sequence length.
- example: 16k input tokens and 1k output tokens.

why this helps -

- long inputs make prefill much heavier than decode.
- colocated serving has to pick one chunk size for both phases.
- if the chunk size is large, decode waits behind long prefill work.
- if the chunk size is small, prefill throughput drops.
- separating prefill and decode lets each phase use a better config.

good use cases -

1. deep research, where many articles/search results are synthesized.
2. rag, where retrieved context can be very large.
3. paper reading / document understanding.
4. agent systems that need to process lots of context.


6. is pd disagg better for big models or small models?

pd disagg gives bigger gains for bigger models.

- for an 8b model, the gap between colocated serving and pd serving may be small.
- for 70b it becomes more useful.
- for 400b+ models the gain can be much larger.

why bigger models benefit more -

- bigger models give more room for different parallelism strategies.
- prefill can use one strategy, like smaller tp with pipeline parallelism.
- decode can use another strategy, like larger tp for lower token latency.
- small models may not have enough work to make these strategies worthwhile.

pd disagg also helps more when the input length is large.

- larger context makes prefill/decode behavior diverge more.
- small input lengths are easier to batch in a colocated system.
- large input lengths make the chunk-size tradeoff harder, so disaggregation helps more.


7. how should traffic shape affect prefill/decode ratio?

for latency-sensitive apps -

- example: siri / realtime assistant.
- users care a lot about fast token streaming.
- use more decode capacity.
- a config like 1 prefill : 2 decode can make sense.

for throughput-oriented apps -

- example: browser agents, rag agents, deep research agents.
- latency can be looser because the work may run in the background.
- inputs are large and prefill-heavy.
- use more prefill capacity.
- a config like 3 prefill : 1 decode can make sense.

so the right ratio comes from traffic and sla, not a fixed rule.
