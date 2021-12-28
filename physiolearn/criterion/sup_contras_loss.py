"""
@Time    : 2021/10/13 16:08
@File    : sup_contras_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupContrasLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(SupContrasLoss, self).__init__()

        self.T = temperature

    def forward(self, feature: torch.Tensor, label: torch.Tensor):
        feature = F.normalize(feature, p=2, dim=-1)

        label = label.view(-1, 1)
        mask = torch.eq(label, label.T).float()

        logits = torch.mm(feature, feature.t().contiguous())
        logits.fill_diagonal_(0)  # mask diagonal
        logits /= self.T

        pos = torch.exp(logits * mask).sum(dim=-1)
        neg = torch.exp(logits * (1 - mask)).sum(dim=-1)

        loss = -torch.log(pos / (pos + neg)).mean()

        return loss
