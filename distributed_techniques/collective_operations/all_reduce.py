import os
import torch
import torch.distributed as dist

def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    tensor = torch.ones(3, device=rank) * (rank + 1)

    print(f"before all_reduce Rank {rank}: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"after all_reduce Rank {rank}: {tensor}")

if __name__ == "__main__":
    example()