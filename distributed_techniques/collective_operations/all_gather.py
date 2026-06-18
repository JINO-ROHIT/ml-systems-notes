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
    
    #for all gather, all the processes sends and receives every tensor

    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor) 

    gathered_tensor = torch.cat(gather_list, dim=0) 
    print("gathered:\n", gathered_tensor)

if __name__ == "__main__":
    example()
