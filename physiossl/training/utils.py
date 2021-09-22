"""
@Time    : 2021/9/21 2:30
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import math
import warnings
import random
from typing import List

import torch
import numpy as np


def setup_seed(seed: int):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr: float, epoch: int, total_epochs: int, lr_schedule: List,
                         method: str = 'lambda'):
    """Decay the learning rate based on schedule"""
    assert method in ['lambda', 'cos']

    if method == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    elif method == 'lambda':  # stepwise lr schedule
        for milestone in lr_schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    else:
        raise ValueError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
