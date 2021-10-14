"""
@Time    : 2021/9/22 11:46
@File    : soft_logit_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLogitLoss(nn.Module):
    def __init__(self):
        super(SoftLogitLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # print(output.shape, target.shape)
        if len(target.shape) == 1:
            target = target.view(-1, 1)
        loss = torch.log(1 + torch.exp(-target * input)).mean()

        return loss
