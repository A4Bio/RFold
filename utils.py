import os
import logging
import numpy as np
import torch
from torch import optim
import random 
import torch.backends.cudnn as cudnn
from collections.abc import Mapping, Sequence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dataset(config):
    from API import load_data
    return load_data(**config)

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    raise TypeError("Can't transfer object type `%s`" % type(obj))