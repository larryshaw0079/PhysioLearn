"""
@Time    : 2021/9/21 2:30
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import math
import random
import warnings
from typing import List, Iterable

import numpy as np
import torch


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
    """Decay the criterion rate based on schedule"""
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


# TODO : to be finished
def logits_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# TODO : to be finished
def mask_accuracy(output: torch.Tensor, target_mask: torch.Tensor, topk: Iterable = (1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)

    zeros = torch.zeros_like(target_mask).long()
    pred_mask = torch.zeros_like(target_mask).long()

    res = []
    for k in range(maxk):
        pred_ = pred[:, k].unsqueeze(1)
        onehot = zeros.scatter(1, pred_, 1)
        pred_mask = onehot + pred_mask  # accumulate
        if k + 1 in topk:
            res.append(((pred_mask * target_mask).sum(1) >= 1).float().mean(0).item())
    return res
