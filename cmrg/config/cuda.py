import torch
import numpy as np
import random
def choonsr_cuda(cuda):
    CUDA = torch.cuda.is_available()  # checking cuda availability
    if CUDA:
        torch.cuda.set_device(cuda)
        print("Choose cuda:{}".format(cuda))
    return CUDA

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # pass


