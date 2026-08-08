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



### 2. kv aware routing

kv cache routing is a technique that nvidia dynamo uses to routes requests between different workers by understanding where kv cache already exists to maximize cache hits. (it also takes other criteria into consideration as well which we will see in a bit)

why kv cache routing is needed

imagine you have 20 gpu worker each with its own kv cache.

for a new incoming request, which worker should the request actually be routed to?

1. if the router sends a request to the wrong worker, it misses a chance to reuse cached kv blocks.
2. if the router only chases cache hits, a few workers can become overloaded.

so the router needs to balance cache reuse and load.

the ideal router has a global view of -
- which workers have matching kv blocks.
- how full each worker's gpu cache is.
- how many requests are waiting on each worker.
- how many active blocks each worker is decoding.



so typically inside a single worker, this is what the kv cache reuse looks like -
1. tokenize the input prompt.
2. split tokens into fixed size blocks, like 16 or 64 tokens.
3. hash each block to create a block id.
4. look up each block in the local kv cache.
5. reuse matching blocks if they exist.
6. allocate memory for missing blocks.
7. evict older / less useful blocks if gpu memory is full.
8. compute key and value tensors for new blocks.
9. store the new kv tensors in cache.

dynamo hooks in this stage and keeps a track of the global state of each kv blocks.
- when a kv block is stored, it emits a kv stored event.
- when a kv block is removed, it emits a kv removed event.
- kvindexer consumes these events and maintains a global prefix tree.

for every new request, kvindexer can return how many blocks match on each worker. this lets the router know which worker has the best prefix reuse.


so is a good cache hit score enough to guarantee fast results? usually this cache match score alone is not enough because if it keeps hitting a single worker, it might get overloaded and end up being the bottleneck.

so dynamo also sends metrics like -

1. `num_requests_waiting` - pending requests.
2. `gpu_cache_usage_perc` - gpu cache usage percentage.
3. `gpu_prefix_cache_hit_rate` - gpu prefix cache hit rate.
4. `request_active_slots` - active request slots.
5. `request_total_slots` - total request slots.
6. `kv_active_blocks` - active kv block count.
7. `kv_total_blocks` - total kv block count.

kvmetricsaggregator receives these metrics and aggregates them by worker.

so a revised flow for how kv router chooses the worker for each request is -

1. receive incoming request tokens.
2. ask kvindexer which workers have matching kv blocks.
3. ask kvmetricsaggregator for current worker metrics.
4. score each worker with a cost function.
5. pick the best worker.
6. sends the request.


whats this router cost function?

the kv routing decision is based on a simple cost function.

logit = kv_overlap_score_weight * potential_prefill_blocks + potential_active_blocks


lower logit is better(less computational cost).

- `potential_prefill_blocks` = the number of kv blocks that would have to be newly computed if this request were sent to that worker.
smaller is better because it means more of the prefix already exists in the kv cache. so a good kv overlap reduces this number.
- `potential_active_blocks` = blocks already active on that worker, so this is a load signal.
- `kv_overlap_score_weight` controls whether we care more about ttft or itl.

if -
1. `kv_overlap_score_weight > 1.0`. the cache overlap term dominates and this prioritizes workers with better kv cache overlap. this automatically results in lower ttft.
2. `kv_overlap_score_weight < 1.0` this spreads decode load more evenly and prefers less busy gpu. this results in a faster ITL.


### sla planner

sla planner is dynamo's autoscaling system.

there are some terms you should know before reading into the sla planner.
1. ttft = time to first token.
2. itl / tpot = inter token latency / time per output token.
3. isl = input sequence length.
4. osl = output sequence length.

what can i do with sla planner?
- scale prefill and decode workers to meet specific ttft and itl requirements.
- it uses performance models/profiling data instead of only reacting after overload.
- also helps to minimize gpu count while meeting the service target.


planner modes

dynamo has multiple planner targets -

1. throughput - this is the default and the recommended mode btw. it sets thresholds on the request queue depth and kv cache utilization and scales up and down depending on this. it is very easy to use since it needs no profiling data or any estimates to get it working. really useful when you want high gpu utilization and good throughput.
2. latency - this is for latency sensitive workloads where we use more aggresive thresholds to scale.
3. load - this uses user defined thresholds to control queue and kv threshold.
4. sla - this is the mode that tries to meet any specific specific targets you might have for ttft and itl. it makes use of a combination of profiling estimates, live metrics and so on to come up with the final prefill decode estimates. 



correction factors

even though profiling gives ideal estimates but production is still quite messy.

some examples are - 
- requests may arrive all at once and queue.
- prefix cache hit rate can reduce actual prefill work.
- small chunked prefills can interfere with decode.
- average isl / osl can hide high variance traffic.

so planner has to take all this into account as well by comparing actual vs expected, something like this.

prefill_correction = actual_ttft / expected_ttft
decode_correction = actual_itl / expected_itl


it then uses these correction factors to make better estimates.

nixl

nixl is actually what makes prefill decode separation efficient since even if we seperate the compute, we still need to find a really efficient way to do kv transfer. if you transfer kv cache the normal way, it will slowly eat away at the ttft. nixl on the other hand transfers kv cache directly between memory locations and is non blocking, which means the gpus can continue useful work while transfer is happening.


lemme close this section with some thoughts on why kv cache does not always become the bottleneck.

consider long inputs for instance -
1. the attention compute is growing roughly n^2.
2. kv cache size on the other hand grows only roughly n.

so as long the context length grows, the compute will always overshadow the kv transfer cost.


some more pointers on how to think about bigger model.
1. bigger models usually mean more attention heads and more compute.
2. besides kv cache growth can be controlled by grouped query attention and mla which reduce the number of kv heads compared to query heads.

this ensures the kv cache size does not necessarily grow linearly with every model dimension and keeps the kv cache transfer bounded enough that gpus can make pd disagg practical.


### fault tolerance

dynamo has fault tolerance for both requests and workers.

main ideas -

1. request migration.
2. request cancellation.
3. graceful shutdown.
4. request rejection / load shedding.
5. health checks.


1. request migration

request migration lets a request continue on another worker when the original worker fails.

two cases -

1. new request migration.
    worker is unreachable during initial connection.
    frontend retries another worker because generation has not started yet.
2. ongoing request migration.
    stream breaks during generation.
    frontend keeps accumulated tokens, reconstructs the request on another worker, and continues from that state.

config -

```bash
python3 -m dynamo.frontend --migration-limit 3
```

default is usually no migration unless configured.


2. request cancellation

request cancellation stops work that no longer matters.

example -

- client disconnects from http stream.
- frontend cancels downstream worker requests.
- worker can stop generating and free compute.

this matters because llm generation is expensive and useless if nobody is reading the response.


3. graceful shutdown

graceful shutdown makes restarts safer.

flow -

1. stop advertising endpoints / stop accepting new requests.
2. keep processing in-flight requests for a grace period.
3. invalidate endpoints.
4. clean up engines, connections, and temp files.
5. exit cleanly so kubernetes can restart or remove the pod.

kubernetes config -

```yaml
extraPodSpec:
  terminationGracePeriodSeconds: 60
```

rule of thumb -

- short requests can use shorter grace periods.
- long generation or high utilization needs longer grace periods.


4. request rejection

request rejection is load shedding.

if workers are overloaded, new requests get rejected instead of making the whole system worse.

prevents -

1. cascading failures.
2. every request getting worse latency.
3. gpu worker oom.

when all workers are busy, frontend can return http 503 so clients retry later.

example thresholds -

```bash
python -m dynamo.frontend \
  --active-decode-blocks-threshold 0.85 \
  --active-prefill-tokens-threshold 10000
```

meaning -

- if decode kv block utilization is too high, worker is busy.
- if active prefill tokens are too high, worker is busy.
- router avoids busy workers.