from recbole.quick_start import run_recbole
import torch.distributed as dist
import os
import torch

def setup(rank=0, world_size=1):
    os.environ['MASTER_ADDR'] = 'localhost'  # Note: Fixed typo here (MASTER_ADDR)
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',  # use 'nccl' for GPU, 'gloo' for CPU
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

setup()

# Define configuration dictionary with user_inter_num_interval
config_dict = {
    'user_inter_num_interval': "[1,1000]",  # Ensures users haven't interacted with ALL items
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},  # Explicitly specify columns
    'neg_sampling': {'uniform': 1}  # Ensure negative sampling is properly configured
}

run_recbole(
    model='Pop', 
    dataset='douban',
    config_dict=config_dict
)

cleanup()