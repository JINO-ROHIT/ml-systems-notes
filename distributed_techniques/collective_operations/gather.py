### torchrun --nproc_per_node=2 script.py

import os
import torch
import torch.distributed as dist

def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    tensor = torch.ones(1, 5, device=rank) * rank
    print(tensor)

    # only destination process or the master process (dst=0) needs memory to collect data
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None  # others just send

    dist.gather(tensor, gather_list=gather_list, dst=0) # every tensor is collected at dst 0

    if rank == 0:
        gathered_tensor = torch.cat(gather_list, dim=0) # change dim to see row vs column variations
        print("gathered:\n", gathered_tensor)

if __name__ == "__main__":
    example()
