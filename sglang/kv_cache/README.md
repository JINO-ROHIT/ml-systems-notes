## sglang kv cache 

the implementation mainly has two structures -
1. memory pool (python/sglang/srt/mem_cache/memory_pool.py)
2. radix cache


#### memory pool

this maps each request to its token locations in the kv cache. it allocates a [size+1, max_context_len] int32 tensor (req_to_token).
main classes -
1. ReqToTokenPool - this maps each request to a request pool
2. MHATokenToKVPool - this holds the physical K/V cache tensors on GPU


#### radix cache

inserts node ino the radix tree and keep splitting depending on the prefix found or if a new node needs to be constructed.


#### do you really understand kv cache?

for every token processed by an llm, it generates two main tensors -
1. key
2. value

these key and value need to be stored somewhere in the gpu memory to serve as a lookup in generating the next token.

sglang stores them in kv cache, not as a whole unit but broken down into pieces called pages. all these pages are registered in a single global page table.


the page table is the global GPU tensor that stores kv location mappings for all currently running requests. it looks like this -

page_table = (max_running_requests, max_seq_len)


the page table does not store the tokens instead it stores physical kv slots -

```
page_table shape = [4, 8] (it has 4 live requests with a max sequence length of 8)
- the values inside are the physical kv slots

                     logical token position
request slot      0    1    2    3    4    5    6    7
              +----+----+----+----+----+----+----+----+
row 0         | .. | .. | .. | .. | .. | .. | .. | .. |  free/unused row
row 1         | 12 | 13 | 14 | 15 | .. | .. | .. | .. |  request B: "a dog ran"
row 2         |  4 |  5 |  6 |  7 | .. | .. | .. | .. |  request A: "the cat sat"
row 3         |  0 |  0 |  0 |  0 | .. | .. | .. | .. |  dummy/padding row
              +----+----+----+----+----+----+----+----+
```

page_table[2, 0] = 4   -> request A token "The" has K/V at physical slot 4
page_table[2, 1] = 5   -> request A token "cat" has K/V at physical slot 5
page_table[2, 2] = 6   -> request A token "sat" has K/V at physical slot 6

page_table[1, 0] = 12  -> request B token "A" has K/V at physical slot 12
page_table[1, 1] = 13  -> request B token "dog" has K/V at physical slot 13
page_table[1, 2] = 14  -> request B token "ran" has K/V at physical slot 14


quick terminology

1. logical means the token's position inside one request's sequence.

```
+------------------+-------+--------+--------+---------+------+
| text             | The   | capital| of     | France  | is   |
+------------------+-------+--------+--------+---------+------+
| logical position | 0     | 1      | 2      | 3       | 4    |
+------------------+-------+--------+--------+---------+------+
```

logical positions are local to a request. request A can have logical token 0, and request B can also have logical token 0.

2. physical is the actual slot in the global GPU KV cache memory.

physical slots are global. there is only one physical slot 8 in the KV cache.

the page table is the one that maps the logical and the physical addresses.

#### kv cache storage

okay perfect, so far so good. now what does the kv cache storage layout look like?

```
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
```

what are these pages? before understanding them, we need to look at a concept called fragmentation.

#### fragmentation

fragmentation in the context exists when memory exists but it is not usable in its current form.

there are two different kinds:

internal fragmentation is when wasted space inside an allocated block/page
external fragmentation is when free space split into pieces that cannot satisfy a larger allocation


### before Paging - contiguous kv allocation

imagine a simpler kv cache allocator where each request needs one contiguous chunk of KV slots.


total KV slots = 16 


after a few requests, memory looks like this -

```text
slot:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |A |A |A |A |..|..|B |B |B |B |..|..|C |C |C |C |
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

free slots in 4, 5, 10, 11


but if a new request needs 4 contiguous slots, it cannot fit since the free slots are not contiguous.

this is called external fragmentation where enough total memory exists, but it is split into unusable pieces


contiguous allocation can also create internal fragmentation if the system reserves more than the request actually uses.


request A reserves max_tokens = 8
request A only uses 5 tokens


```text
+----+----+----+----+----+----------+----------+----------+
| A0 | A1 | A2 | A3 | A4 | reserved | reserved | reserved |
+----+----+----+----+----+----------+----------+----------+
```

the last 3 slots are inside request A's allocation, so other requests cannot use them. this is called internal fragmentation.

### after paging

sglang avoids external fragmentation by allocating fixed-size pages.

let's say your page_size = 4

KV memory is now split like this - 

```text
physical page 0        physical page 1        physical page 2        physical page 3
+---+---+---+---+    +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
| 0 | 1 | 2 | 3 |    | 4 | 5 | 6 | 7 |    | 8 | 9 |10 |11 |    |12 |13 |14 |15 |
+---+---+---+---+    +---+---+---+---+    +---+---+---+---+    +---+---+---+---+
```

free pages are tracked by page starts.
free_slots = [0, 4, 8, 12]
If pages 1 and 3 are free, free_slots = [4, 12]

they are not contiguous in physical memory, but a request needing 2 pages can still use them:

allocated page starts = [4, 12]
expanded token slots  = [4, 5, 6, 7, 12, 13, 14, 15]

the page table hides the non-contiguous physical layout.

```
logical token:   0  1  2  3   4   5   6   7
physical slot:   4  5  6  7  12  13  14  15
```

this way, paging reduces external fragmentation because the request does not need one contiguous physical range.

the remaining main waste is internal fragmentation inside the last page.

for instance
page_size = 4
request length = 6


the request needs:

```text
ceil(6 / 4) = 2 pages
```

capacity - 2 pages * 4 slots = 8 slots

but we need only 6 slots, so 2 slots are wasted


```text
request A owns two pages

page 0                 page 1
+----+----+----+----+ +----+----+----------+----------+
| A0 | A1 | A2 | A3 | | A4 | A5 | reserved | reserved |
+----+----+----+----+ +----+----+----------+----------+
```

Those reserved slots are internal fragmentation. They cannot be used by another request because this allocator owns/frees whole pages.


so should i pick a smaller page size or a larger page size?

smaller page_size
1. less internal fragmentation
2. more page table entries and more page-management overhead

larger page_size
1. fewer page-table entries and often better locality
2. more internal fragmentation in partially filled final pages

#### how is a page allocated

for every incoming batch of requests, the scheduler is in charge of allocating pages but looking at two different fields.

cached_len - how many tokens are already in the kv cache
device_len - how many tokens now need to be stored in the cache

the allocation logic looks like this -

first_page = div_ceil(req.cached_len, self.page_size)
last_page = div_ceil(req.device_len, self.page_size)

If last_page > first_page, then the request needs more pages.


for instance, consider -
page_size = 4
cached_len = 0
device_len = 6

first_page = ceil(0 / 4) = 0
last_page  = ceil(6 / 4) = 2

the request needs 2 pages.


#### partial pages

okay this is cool, what about partial pages? can they be used by new requests?

no they cannot! these remaining slots for the partial pages are usually for the newer tokens generated during decode to be stored.

and when the tokens generated exceeded the current allocated slot in the page, a newer page is allocated for the next tokens to be stored.