### collective operations in nccl

this is a collection of the main collective operations in nccl that i used for practice.

1. `gather.py` 
    - each gpu has its own tensor
    - all tensors are collected (concatenated) onto a single gpu (root)
    - other gpus do not receive the full result
2. `all_gather.py` - here, instead of a single gpu having all the data copy, all the gpus have all the copies of each other.
3. `reduce.py` 
    - all gpus have some tensors
    - a reduce op (sum, mean etc) is applied element wise
    - result is stored on one GPU (root)
4. `all_reduce.py` - here, all the gpus have the reduced copy
5. `scatter.py` 
    - one gpu has a large tensor split into N chunks
    - each gpu receives one chunk
    - kinda opposite to gather
6. `reduce_scatter.py`
    - all gpus start with full tensors
    - a reduce op is applied across gpus
    - the result is split, and each gpu gets only its chunk
    - same as reduce + scatter