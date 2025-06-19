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

# setup()

# # Define configuration dictionary with user_inter_num_interval
# config_dict = {
#     'user_inter_num_interval': "[1,1000]",  # Ensures users haven't interacted with ALL items
#     'load_col': {'inter': ['user_id', 'item_id', 'rating']},  # Explicitly specify columns
#     'neg_sampling': {'uniform': 1}  # Ensure negative sampling is properly configured
# }

from recbole.quick_start import run_recbole

parameter_dict = {
    'gpu_id': 0,
    'epochs': 50,
    'train_batch_size': 512,
    'eval_batch_size': 512,
    'metrics': ['Hit', 'NDCG'],  # only ranking metrics
    'topk': [10, 100],
    'valid_metric': 'NDCG@10',
    'eval_setting': 'TO_LS,full',
    'seed': 42,
}

# parameter_dict = {
#    'train_neg_sample_args': None,
# }

run_recbole(model='NeuMF', dataset='ml-100k', config_dict=parameter_dict)

# cleanup()