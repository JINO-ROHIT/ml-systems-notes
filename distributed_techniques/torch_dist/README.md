## torch distributed

`--nnodes` - this is the number of machines/servers you have
`--nproc_per_node` - this is the number of gpus you have within the machine.


```bash
# 2 machines, 8 GPUs each = 16 total processes
torchrun --nnodes=2 --nproc_per_node=8 train.py
```

### meta device

this is an abstract device that records metadata but no data. this means you dont need to load tensors on cpu/gpu but check transofrmations, analysis on the
tensors etc without actually spending time on loading stuff, no OOMs etc.


```python
import torch
from torch import nn

model = nn.Linear(10, 5).to("meta")
x = torch.randn(3, 10).to("meta")
out = model(x) # no memory allocated
print(out.shape) # you get torch.Size([3, 5])
```

### process group

the main crux of doing distributed training is a way for processes to find and talk to each other. you do this using process group.
also let say we have 4 gpus, we need gpu 1 and 3 to talk, and gpu 2 and 4 to talk to each other, and not with other. process groups help you do this.

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
# all processes now belong to the default world group

# only let ranks 0 and 1 talk to each other
group_01 = dist.new_group([0, 1])
# only let ranks 2 and 3 talk to each other
group_23 = dist.new_group([2, 3])
```


### device mesh

a deviceMesh is essentially a structured way to create and manage many process groups. as you scale more and more gpus, using process groups alone gets quite complicated.


```python
from torch.distributed.device_mesh import init_device_mesh

# create a 2D mesh: 2 nodes × 4 GPUs per node
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("pp", "tp"))

# automatically creates sub groups:
pp_group = mesh["pp"]       # process group for pipeline parallelism
tp_group = mesh["tp"]       # process group for tensor parallelism
```

### dtensor

the native tensor type used for distributed training.

you can shard, replicate and partial ops.


```python
from torch.distributed.tensor import DTensor, Shard, Replicate, Partial

mesh = init_device_mesh("cuda", (4,))

# shard along dim 0 across devices
dt = DTensor.from_local(local_tensor, mesh, [Shard(0)])

# replicate across all devices
dt = DTensor.from_local(local_tensor, mesh, [Replicate()])
```
