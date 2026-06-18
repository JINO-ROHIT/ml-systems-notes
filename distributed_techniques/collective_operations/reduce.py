import os
import torch
import torch.distributed as dist

def example():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    tensor = torch.ones(3, device=rank) * (rank + 1)
    print(f"rank {rank} tensor before reduce: {tensor}")

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    print(f"rank {rank} tensor after reduce: {tensor}")

if __name__ == "__main__":
    example()