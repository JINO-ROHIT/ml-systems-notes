# reference - https://jax-ml.github.io/scaling-book/roofline/

llms are all but lots of matrix multiplications with some other operations mixed in between.

1. computation - how long these can take to compute , lets call it Tmath = computation flops / accelerator flops/s  

2. communication on the same chip - usually these tensors on the gpu, they need to be moved across the HBM to the compute cores. this bandwith is called the HBM bandwith. 

3. communication across chips - what if we distribute these tensor across multiple chips. you need something like pcie, nvlink etc to move them.

lets call it Tcomm = communication bytes / memory bandwith bytes

if we are able to overlap the computation and communication costs, in that case -
1. the lower bound is max(computation, communication)
2. upper bound is computation + communication

when Tmath > Tcomm = compute bound
when Tcomm > Tmath = memory bound

to determine this, we need arithmetic intensity(AI)

AI = computation flops / communication bytes

if AI is high, we spend most of our time doing copmputation, using most of our flops which is good. the opposite is when time gets eaten away in communication.

lets take dot product for example of two vectors in bf16(2 bytes)

- we have N multiplications and N - 1 additions
- we load vector A(2N bytes) and then vector B(2N bytes) and then write back one single final element(2)
AI = N + N - 1 / 2N + 2N + 2 ~= 1 / 2

this does 0.5 float point operations for each byte loaded.


lets do for matmul now, 
- X is bf16 [B, D]
- Y is bf16 [D, F]
- Z is bf16 [B, F]

AI = 2BDF / 2BD + 2DF + 2BF 

if we assume B batch size is small compared to D and F, then we can simplify to

AI = BDF / DF ~= B

so AI in this case depends on the batch size, so at what batch size tokens do we become compute bound.

AI(matmul) > AI(accelator for tp for example)
= B > 240 (flops/bytes moved for that machines)

this means batch size tokens should be atleast 240.

what about communication across multiple chips?

lets say we now do the matmul across 2 chips. this means we multiply half of the matrix and then send over the partial matrices to one chip for the final accumulation.

assume we split the reduction dimension D across two chips:

- chip 0 computes `X[:, :D/2] @ Y[:D/2, :] = Z0[B, F]`
- chip 1 computes `X[:, D/2:] @ Y[D/2:, :] = Z1[B, F]`
- final result is `Z = Z0 + Z1`

the total flops are still:

AI numerator = 2BDF

but now one chip needs to send its partial output to the other chip for the final accumulation. the partial output is bf16 `[B, F]`, so the extra inter-chip communication is:

communication bytes = 2BF

so if we include both HBM traffic and inter-chip communication:

AI(matmul on 2 chips) = 2BDF / (2BD + 2DF + 2BF + 2BF)
= 2BDF / (2BD + 2DF + 4BF)
= BDF / (BD + DF + 2BF)

if B is small compared to D and F, then the denominator is still dominated by DF.
AI(matmul on 2 chips) ~= BDF / DF ~= B

so for HBM arithmetic intensity, the compute-bound threshold is still B > AI(accelerator)

but this hides the inter-chip communication question. for inter-chip communication alone -
AI(inter-chip) = flops / inter-chip bytes
= 2BDF / 2BF
= D

so with this D-split, increasing B helps the HBM roofline, but it does not make the inter-chip reduction more favorable. the inter-chip AI depends on D.

if we send partials in fp32 instead of bf16, then communication bytes become `4BF`
AI(inter-chip, fp32 partials) = 2BDF / 4BF = D / 2


### exercises


**Question 1 [int8 matmul]:** Say we want to do the matmul $X[B,D] \cdot Y[D,F] \to Z[B,F]$ in int8 precision (1 byte per parameter) instead of bfloat16 (2 bytes per parameter) since TPUs/GPUs can do matmuls faster in lower precision.

- How many bytes need to be loaded from memory? How many need to be written back to memory?
- How many total OPs are performed?
- What is the arithmetic intensity?
- What is a roofline estimate for $T_{\text{math}}$ and $T_{\text{comms}}$? What are reasonable upper and lower bounds for the runtime of the whole operation?

Assume our HBM bandwidth is `8.2e11 bytes/s` and our int8 peak OPs/s is `3.94e14` (about 2x bfloat16).


1. 

- bytes to be loaded = BD + DF
- bytes to write = BF

2. ops performed = 2BDF

3. arithmetic intensity = 2BDF / (BD + DF + BF)

if we assume B(batch size) is relative smaller compared to D and F, then it becomes = 2BDF / DF = 2B

to become compute bound when 2B > (3.94e14 / 8.21e11) = B > 240
when your batch size becomes bigger than 240. 

4. 
- Tmath = 2BDF / 3.94e14
- Tcomm = (BD + DF + BF) / 8.2e11 

lower bound is max(Tmath, Tcomm)
upper bound is Tmath + Tcomm



**Question 2 [int8 + bf16 matmul]:** In practice we often do different weight vs. activation quantization, so we might store our weights in very low precision but keep activations (and compute) in a higher precision. Say we want to quantize our weights in int8 but keep activations (and compute) in bfloat16. At what batch size do we become compute bound? Assume 1.97e14 bfloat16 FLOPs/s.

Hint: this means specifically bf16$[B, D]$ * int8$[D, F]$ → bf16$[B, F]$ where $B$ is the batch size.

arithmetic intensity = 2BDF / (2BD + DF + 2BF)
same assumption = 2B > 240 or B > 120




**Question 3:** Taking the setup from Question 2, make a roofline plot of peak FLOPs/s vs. batch size $B$ for $F=D=4096$ and $F=D=1024$. Use the exact number of bytes loaded, not an approximation.

```
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

**Question 4:** What if we wanted to perform int8$[B,D]$ $\cdot$ int8$[B,D,F]$ → int8$[B,F]$ where we imagine having a different matrix for each batch element. What is the arithmetic intensity of this operation?

arithmetic intensity = 2BBDF / (BD + BDF + BF)

*TO-DO - this is wrong and need to understand this
