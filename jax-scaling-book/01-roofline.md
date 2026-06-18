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
