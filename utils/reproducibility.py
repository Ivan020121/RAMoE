"""Original random-seed and cuDNN settings."""
import random

import numpy as np
import torch
from torch.backends import cudnn


def fix_random_seed(seed=42, disable_cuda_random=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if disable_cuda_random:
        cudnn.benchmark = False
        cudnn.deterministic = True
