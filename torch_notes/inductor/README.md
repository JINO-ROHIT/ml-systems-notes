### torchinductor

## phase 1 - understanding the IR

in this phase, the fx graph produced by torchdynamo is converted into **inductor ir**.

```python
@torch.compile
def f(x):
    b = torch.floor(x) + torch.ceil(x)
    c = b.sum(dim=-1)
    d = c + 1
    return d
```

becomes

```
x
│
├── floor
├── ceil
│
└── add
     │
     sum
      │
     add(1)
      │
    output
```

graphlowering walks over every fx node and converts it into an inductor ir node.

---

## 1. placeholders

the placeholder `x` becomes

```python
TensorBox(
    StorageBox(
        InputBuffer(
            name="arg0_1",
            layout=FixedLayout(
                device="cuda:0",
                dtype=torch.float32,
                size=[32,512,1024],
                stride=[524288,1024,1]
            )
        )
    )
)
```


```python
InputBuffer(...)
```

it stores metadata about the input tensor but no computation happens here

* device
* dtype
* shape
* strides
* layout


## 2. pointwise operations

the fx node

```python
floor(x)
```

becomes

```python
TensorBox(
 StorageBox(
  Pointwise(
    'cuda:0',
    torch.float32,

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.floor(tmp0)

        return tmp1

    ranges=[32,512,1024]
)))
```

inductor stores a **recipe** for computing one output element but never actually computes it


`ceil(x)` produces almost identical ir

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.ceil(tmp0)

        return tmp1
)))
```


## 3. pointwise fusion

the next fx node

```python
add(floor(x), ceil(x))
```

becomes

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1,i2 = index

        tmp0 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp1 = ops.floor(tmp0)

        tmp2 = ops.load(
            arg0_1,
            i2 + 1024*i1 + 524288*i0
        )

        tmp3 = ops.ceil(tmp2)

        tmp4 = tmp1 + tmp3

        return tmp4
)))
```


the previous individual `Pointwise` nodes disappeared but instead, their `inner_fn` functions were copied into this new one.

instead of

```
floor --> temporary tensor --> ceil --> temporary tensor --> add
```

inductor now stores

```
load --> floor --> load --> ceil --> add
```

as a single recipe, this is called fusion!


## 4. reduction

now we encounter

```python
sum(dim=-1)
```

this cannot be represented using pointwise ir because each output element depends on many input elements.


so inductor creates

```python
TensorBox(
 StorageBox(
  ComputedBuffer(

    name="buf0",

    data=Reduction(

      ranges=[32,512],
      reduction_ranges=[1024],

      def inner_fn(index,rindex):

          i0,i1=index

          r0=rindex

          tmp0 = ops.load(
              arg0_1,
              r0 + 1024*i1 + 524288*i0
          )

          tmp1 = ops.floor(tmp0)

          tmp2 = ops.load(
              arg0_1,
              r0 + 1024*i1 + 524288*i0
          )

          tmp3 = ops.ceil(tmp2)

          tmp4 = tmp1 + tmp3

          return tmp4
)))
```


conceptually this is like

```python
for i0 in range(32):
    for i1 in range(512):

        total = 0

        for r0 in range(1024):

            total += floor(x[i0,i1,r0]) + ceil(x[i0,i1,r0])

        output[i0,i1] = total
```

notice that the floor, ceil and add computations appear again but this is **not recomputation**.

because those tensors never existed. only their recipes existed.

reduction simply copies those recipes into its own computation.
---

## 5. computedbuffer

the reduction result is wrapped in

```python
ComputedBuffer(
    name="buf0"
)
```

this means that the result of this reduction is now a logical tensor that future operations can read.

it does **not** necessarily mean memory has already been allocated and whether it lives in registers, shared memory or global memory is decided later by the scheduler.


## 6. remaining pointwise operations

the next operation

```python
d = c + 1
```

becomes

```python
TensorBox(
 StorageBox(
  Pointwise(

    def inner_fn(index):

        i0,i1 = index

        tmp0 = ops.load(
            buf0,
            i1 + 512*i0
        )

        tmp1 = ops.constant(
            1,
            torch.float32
        )

        tmp2 = tmp0 + tmp1

        return tmp2
)))
```

unlike previous pointwise nodes,

this one loads from

```
buf0
```

instead of the input tensor.


## 7. output

finally the output node wraps the previous pointwise computation inside another

```python
ComputedBuffer(
    name="buf1"
)
```

this is the tensor returned by the compiled function.

we have only looked at few ops but pytorch has 1000s of other ops and they have their own IRs.


## phase 2 - how lowering works

now that we understand what inductor ir looks like, lets look at how the fx graph actually gets lowered into that ir.

### 2.1 the lowering registry

every aten op has a lowering function registered via `@register_lowering`. lets look at `ceil`:

```python
@register_lowering(aten.ceil)
def ceil(x):
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("ceil")
    return make_pointwise(fn)(x)
```

`x` here is an inductor ir node (could be `InputBuffer`, `ComputedBuffer`, or even an unmaterialized `Pointwise`).


`make_pointwise` is a helper that builds pointwise ir nodes. heres a simplified version:

```python
def make_pointwise(fn, ...):
    def inner(*inputs: List[TensorBox], alpha=None):
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()

        def inner_fn(index):
            return fn(*[load(index) for load in loaders])

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )
    return inner
```

notice the three layers of nesting:
1. `make_pointwise(fn)` - configures the math function
2. `inner(*inputs)` - receives inductor ir nodes, builds `inner_fn` but **does not call it**
3. `inner_fn(index)` - the per-element recipe, called later during codegen

`inner` only wraps `inner_fn` into a `Pointwise` ir node. the actual call to `inner_fn` happens during codegen.

to see the difference, trace what happens for `ceil(x)`:

```
step 1: ceil(x) is called during lowering
        x is an InputBuffer ir node (shape [32,512,1024])

step 2: ceil calls make_pointwise(ops_wrapper("ceil"))(x)
        └── make_pointwise(fn) returns inner (the closure)
        └── inner(x) is called now

step 3: inside inner(x):
        loaders = [x.make_loader()]        # InputBuffer.make_loader()
        ranges = [32, 512, 1024]

        def inner_fn(index):               # defined but NOT called
            return ops.ceil(loaders[0](index))

        return Pointwise.create(           # returns IR node, not a value
            inner_fn=inner_fn,             # inner_fn stored as a recipe
            ranges=[32,512,1024]
        )

step 4: result: a Pointwise IR node (the recipe). no computation happened.
```

`inner` builds the ir graph. `inner_fn` is inspected by codegen to emit the actual kernel code. neither of them actually runs `ceil` on data - that happens when the compiled triton kernel executes on the gpu.

### 2.3 how loaders work

each ir type has its own `make_loader()`. this is the key to understanding fusion:

- **`InputBuffer.make_loader()`** - returns a function that calls `ops.load(buf, offset)`. this reads data directly from the input tensor.

- **`Pointwise.make_loader()`** - returns its own `inner_fn`! so when you call `load(index)` on a pointwise node, you get back its computation recipe.

this is how fusion happens. when `add(floor, ceil)` is lowered:

1. `make_pointwise(add_fn)(floor_ir, ceil_ir)` is called
2. `floor_ir.make_loader()` returns `floor`'s `inner_fn` (load to floor)
3. `ceil_ir.make_loader()` returns `ceil`'s `inner_fn` (load to ceil)
4. the new `inner_fn` calls both loaders, then adds:

```python
def inner_fn(index):
    floor_val = floor_loader(index)  # floor's inner_fn
    ceil_val = ceil_loader(index)    # ceil's inner_fn
    return floor_val + ceil_val
```

## phase 3 - from inner_fn to triton code

weve seen that `inner_fn` is just a python function that defines the per-element computation. now lets see how inductor turns that function into actual triton kernel code.

### 3.1 pretty printing inner_fn

how does inductor print the `inner_fn` code we saw in phase 1? it uses `KernelFormatterHandler` to run `inner_fn` with a fake ops handler that captures each op as a string:

```python
class KernelFormatterHandler:
    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None):
        with V.set_ops_handler(formatter):
            result = ir_fn(*args)
            return formatter.getvalue(result)
```

when `inner_fn` runs under this handler, every `ops.load`, `ops.floor`, etc. generates a string like `"tl.load(in_ptr0 + (x0), None)"` instead of actually computing. each op assigns its result to a temp variable:

```python
def inner(*args, **kwargs):
    line = getattr(self.parent_handler, name)(*args, **kwargs)
    varname = f"tmp{next(self.var_counter)}"
    self.output.writeline(f"{varname} = {line}")
    return varname  # next op uses this varname as input
```

so `inner_fn` gets traced by running it - the index values dont matter, only the **structure** of ops matters.

### 3.2 the actual codegen path

for real triton codegen, inductor does something more involved:

```
inner_fn
    
LoopBodyBlock converts inner_fn to an FX graph
    
append ops.store / ops.store_reduction at the end
    
TritonKernel walks the FX graph and emits Triton code
```

### 3.3 pointwise kernels

take the simplest case:

```python
@torch.compile
def fa(x):
    a = torch.floor(x) + torch.ceil(x)
    return a
```

this produces a flat pointwise triton kernel:

```python
@pointwise(size_hints=[16777216], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.math.floor(tmp0)
    tmp2 = tl.math.ceil(tmp0)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
```

notice two things:

1. **dimensions are merged** - the 3d shape [32,512,1024] becomes 1d [16777216]. inductor merges contiguous dimensions to simplify loop logic.

2. **xblock autotuning** - `@pointwise` provides block sizes (usually 1024 or 512) and benchmarks them. the kernel is a single flat loop over `xnumel` elements.

### 3.4 reduction kernels

reductions produce different triton code depending on the reduction dimension size.

**case 1: reduction dim is medium-sized (rnumel=1024)**

the entire reduction dimension fits in one RBLOCK. inductor uses `persistent_reduction`:

```python
@persistent_reduction(size_hints=[16384, 1024], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1          # one row per program
    RBLOCK: tl.constexpr = 1024       # entire reduction dim
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    rindex = tl.arange(0, RBLOCK)[:]
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), rmask, other=0)
    tmp1 = tl.math.floor(tmp0)
    tmp2 = tl.math.ceil(tmp0)
    tmp3 = tmp1 + tmp2
    tmp7 = tl.sum(tmp3, 0)             # reduce over RBLOCK
    tl.store(out_ptr0 + (x0), tmp7, None)
```

xblock=1 because each thread block handles one row. the entire row is loaded, the computation runs, then tl.sum reduces it.

**case 2: reduction dim is very small (rnumel=16)**

xblock can be larger since the reduction is cheap:

```python
@persistent_reduction(size_hints=[16384, 16], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # 2d grid
    rindex = tl.arange(0, RBLOCK)[None, :]
    ...
    tmp7 = tl.sum(tmp6, 1)[:, None]    # reduce over RBLOCK per XBLOCK row
    tl.store(out_ptr0 + (x0), tmp7, None)
```

here xblock is autotuned from [1,8,32,128] since the reduction is cheap enough that multiple rows can share a thread block.

**case 3: reduction dim is large (rnumel=32768)**

the entire row doesnt fit in one RBLOCK. inductor uses `reduction` with a for loop:

```python
@reduction(size_hints=[16384, 32768], ...)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        tmp0 = tl.load(in_ptr0 + (r1 + 32768*x0), rmask, other=0)
        tmp1 = tl.math.floor(tmp0)
        tmp2 = tl.math.ceil(tmp0)
        tmp3 = tmp1 + tmp2
        tmp6 = _tmp5 + tmp3              # accumulate
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]    # final reduce
    tl.store(out_ptr0 + (x0), tmp5, None)
```

each iteration loads [xblock, rblock] elements, accumulates, and after the loop a final `tl.sum` reduces. rblock is autotuned.

### 3.5 cse - common subexpression elimination

in the ir, `inner_fn` loads the same input twice (once for floor, once for ceil):

```python
tmp0 = ops.load(arg0_1, offset)   # for floor
tmp2 = ops.load(arg0_1, offset)   # for ceil
```

but in the generated triton code, it loads only once:

```python
tmp0 = tl.load(in_ptr0 + (x0), None)
tmp1 = tl.math.floor(tmp0)
tmp2 = tl.math.ceil(tmp0)          # reuses tmp0, no second load
```

this happens because inductor's `CSE` class deduplicates identical expressions:

```python
class CSE:
    def generate(self, buffer, expr):
        var = self.cache.get(expr)
        if not var:                  # first time: create new variable
            var = self.newvar()
            self.cache[expr] = var
            buffer.writeline(f"{var} = {expr}")
        return var                   # cache hit: reuse previous variable
```

the two loads have the same address expression, so the second one returns the same `tmp0` variable.

### 3.6 the fx graph intermediate

before generating triton code, inductor converts `inner_fn` into a small FX graph. the graph for `fa` (pointwise) looks like:

```
[get_index, load, floor, get_index, load, ceil, add, get_index, store, output]
```

for `fb` (reduction), it has extra nodes:

```
[get_index, load, floor, get_index, load, ceil, add, reduction, get_index, store_reduction, output]
```

the `store` / `store_reduction` nodes are appended by `ComputedBuffer.get_store_function()`:

```python
class Pointwise(Loops):
    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))

class Reduction(Loops):
    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        value = ops.reduction(self.dtype, self.src_dtype,
                              self.reduction_type, self.inner_fn(vars, reduction_vars))
        return ops.store_reduction(output_name, indexer(vars), value)
```

the FX graph is then consumed by `TritonKernel.codegen_body()`, which splits the code into:

- **indexing_code** - index calculations (r1 = rindex, x0 = xindex)
- **loads** - tl.load statements
- **compute** - the math ops (floor, ceil, add)
- **stores** - tl.store statements
- **suffix** - final reduction (tl.sum)

for reduction kernels with large dims, `codegen_body` wraps loads/compute/stores in a `for roffset` loop. for persistent reductions, everything is flat.
