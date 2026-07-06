# hand rolling flops for an llm

deriving the compute cost of an llm forward pass from first principles.


what is considered a flop?

a **flop** (floating-point operation) is one multiply or one add. a fused multiply-add (fma) does both in one instruction so counts as **2 flops**.

example - 

```
a * b + c    = 2 flops (1 multiply + 1 add)
```

nice, lets start from matrix multiplication and then scale to each layer present in an llm.


## 1. matrix multiplication

suppose we multiply

```
a (m x k)  *  b (k x n)  =  c (m x n)
```

### cost of one output element

each element c[i][j] is a dot product of row i of a and column j of b:

- **k multiplications** (one per pair)
- **k − 1 additions** 

so per element:

```
flops_per_element = k + (k − 1) = 2k − 1
```

### now for all output elements

the output matrix has m x n elements, so:

```
flops_exact = m * n * (2k − 1)
```

you can approximate it to -

```
flops_matmul ≈ 2 * m * k * n
```

great! lets do linear layers now

## 2. linear layer

a linear layer `y = xw` is just a matrix multiplication.

### shapes

- input: `x = (b * s * d)` where b = batch, s = sequence length, d = hidden dim
- weight: `w = (d * d_out)`
- output: `y = (b * s * d_out)`

### this is essential still a matrix multiplication

we flatten the first two dimensions: `(b*s) * d` and multiply by `d * d_out`.

```
m = b*s
k = d
n = d_out
```

### flops

```
flops_linear = 2 * (b*s) * d * d_out
```

when `d_out = d` (most transformer projections):

```
flops_linear = 2 * b * s * d^2
```

### example

b = 1, s = 2048, d = 4096:

```
flops = 2 * 1 * 2048 * 4096 * 4096 ≈ 6.87 * 10¹⁰  (68.7 gflops)
```

---

## 3. attention

a standard attention head does:

```
q = x * w_q       k = x * w_k       v = x * w_v           (3 projections)
s = q * k^T                                                       (scores)
p = softmax(s)                                                     (per-row)
o = p * v                                                        (weighted sum)
y = o * w_o                                                    (output projection)
```

let's break it down. assume:

- b = batch
- s = sequence length
- d = hidden dimension
- h = number of heads
- d_h = d / h = head dimension

### for qkv projections (3 linear layers)

each is a linear layer: (input d --> output d).

```
flops_q_proj  = 2 * b * s * d * d = 2 * b * s * d^2
flops_k_proj  = 2 * b * s * d^2
flops_v_proj  = 2 * b * s * d^2
```

sum:

```
flops_qkv = 6 * b * s * d^2
```

### for attention scores (q * k^T)

we can view this per head, or in total.

**per head:** q has shape `(b*s, d_h)`, k^T has shape `(d_h, s)`. the result is `(b*s, s)`.

```
m = b*s
k = d_h
n = s

flops_per_head = 2 * b * s * d_h * s = 2 * b * s^2 * d_h
```

**all h heads:**

```
flops_scores = h * 2 * b * s^2 * d_h = 2 * b * s^2 * (h * d_h) = 2 * b * s^2 * d
```

### for softmax

softmax operates on each row of the score matrix. a row of length s requires:

- **max reduction**: s comparisons (find max to subtract for numerical stability)
- **subtract + exp**: s fmas (exponential counts as ~1 flop by convention)
- **sum reduction**: s − 1 additions
- **divide**: s divisions

per row ≈ **4s** flops (rough estimate).

there are b*s rows per head, times h heads:

```
flops_softmax ≈ 4 * b * s * h * s = 4 * b * s^2 * h
```

in terms of d:

```
flops_softmax ≈ 4 * b * s^2 * d / d_h = (4 / d_h) * b * s^2 * d
```

for typical d_h = 64–128, this is small relative to the matmuls. many analyses drop it.

### for attention output (p * v)

this is another matmul. per head:

- p: `(b*s, s)`
- v: `(s, d_h)` (v was originally `(b*s, d)`, reshaped to `(b*s, h, d_h)`)
- result: `(b*s, d_h)`

```
m = b*s
k = s
n = d_h

flops_per_head = 2 * b * s * s * d_h = 2 * b * s^2 * d_h
```

all h heads:

```
flops_attn_out = 2 * b * s^2 * d
```

### for output projection (o * w_o)


```
flops_output_proj = 2 * b * s * d^2
```

### total attention flops

```
flops_attention =  6 * b * s * d^2      (qkv projections)
                 + 2 * b * s^2 * d      (scores)
                 + 2 * b * s^2 * d      (weighted sum)
                 + 2 * b * s * d^2      (output projection)

                 = 8 * b * s * d^2  +  4 * b * s^2 * d
```

## 4. feed-forward (ffn) layer

a standard ffn has two projections with an activation in between:

```
z = activation(x * w_up)           expand d to 4d
y = z * w_down                     project 4d to d
```

for a gated ffn (glu variants like swiglu used in llama, mistral, etc.):

```
z1 = x * w_gate     (d to 4d)
z2 = x * w_up       (d to 4d)
z  = activation(z1) * z2           elementwise
y  = z * w_down     (4d to d)
```

### swiglu flops

three matmuls, each `d to 4d` or `4d to d`:

```
flops_ffn_gate = 2 * b * s * d * 4d = 8 * b * s * d^2
flops_ffn_up   = 2 * b * s * d * 4d = 8 * b * s * d^2
flops_ffn_down = 2 * b * s * 4d * d = 8 * b * s * d^2
```

total:

```
flops_ffn = 24 * b * s * d^2
```


## 6. one transformer layer

a standard decoder layer = attention + ffn

```
flops_layer = flops_attention + flops_ffn

             = (8 * b * s * d^2 + 4 * b * s^2 * d) + (24 * b * s * d^2)

             = 32 * b * s * d^2 + 4 * b * s^2 * d
```

## 7. full model forward pass

for l layers:

```
flops_forward = l * (32 * b * s * d^2 + 4 * b * s^2 * d)

              = 32 * l * b * s * d^2  +  4 * l * b * s^2 * d
```

### embedding layer

the embedding lookup has 0 flops.

the final lm head (unembedding) is a linear `d to vocab_size`. in many models the lm head shares weights with the embedding. this adds:

```
flops_lm_head = 2 * b * s * d * vocab_size
```

for large vocabularies (llama 128k), this would be quite significant comparable to a few transformer layers.


## per token flops

often we care about flops *per token* (to compare with memory bandwidth or compute roofline).

total tokens processed in the forward pass = `b * s`.

```
flops_per_token = flops_forward / (b * s)

                = 32 * l * d^2  +  4 * l * s * d
```

two terms:

- **`32 * l * d^2`**: per-token compute, independent of sequence length. this is ~2 * the total non-embedding parameter count (since each param is used once in a matmul with `m = 1` token).
- **`4 * l * s * d`**: attention overhead per token, grows linearly with s.

### numeric example

**llama 3 8b** (l = 32, d = 4096):

```
flops_per_token (parameter term) = 32 * 32 * 4096^2 ≈ 17.2 * 10⁹ = 17.2 gflops/token
```

the total non-embedding parameters are ~8 * 10⁹, and we got ~17 gflops/token. that's `~2.15 * parameter_count` because the flop/param ratio of a `d * d` matmul is:

```
2 * d * d * d   (for the matmul: m=k=n=d, flops = 2*d*d*d = 2d³)
= 2 * d          per parameter (since there are d^2 params)
```

each parameter in a square linear layer is involved in 2*d flops per forward pass.

---

## 9. backward pass

rough rule of thumb: **backward is ~2* forward** for compute.

```
flops_backward ~ 2 * flops_forward
```

### total training flops per token

```
flops_training_per_token = flops_forward + flops_backward
                         ~ 3 * flops_forward_per_token

                         ~ 3 * (32 * l * d^2 + 4 * l * s * d)

                         ~ 96 * l * d^2 + 12 * l * s * d
```

## summary formulas

| component | flops |
|-----------|-------|
| linear layer (d to d_out) | `2 * b * s * d * d_out` |
| qkv projections (total) | `6 * b * s * d^2` |
| attention scores | `2 * b * s^2 * d` |
| attention weighted sum | `2 * b * s^2 * d` |
| output projection | `2 * b * s * d^2` |
| total attention | `8 * b * s * d^2 + 4 * b * s^2 * d` |
| swiglu ffn | `24 * b * s * d^2` |
| **one transformer layer** | `32 * b * s * d^2 + 4 * b * s^2 * d` |
| **full model (l layers), forward** | `32 * l * b * s * d^2 + 4 * l * b * s^2 * d` |
| **full model, backward** | ~2 * forward |
| **full model, training (forward + backward)** | ~3 * forward |
