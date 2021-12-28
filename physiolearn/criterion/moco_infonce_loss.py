"""
@Time    : 2021/10/13 16:10
@File    : moco_infonce_loss.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MocoInfoNCELoss(nn.Module):
    def __init__(self, T: float):
        super(MocoInfoNCELoss, self).__init__()

        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_q: torch.Tensor, z_k: torch.Tensor, queue: torch.Tensor):
        assert z_q.shape == z_k.shape

        B, *_ = z_q.shape  ## (batch, *, feature_dim)
        feature_dims = z_q.shape[:-1]  ## without the representation dim
        dims_prod = np.prod(feature_dims)

        z_q = F.normalize(z_q, p=2, dim=-1)
        z_k = F.normalize(z_k, p=2, dim=-1)

        logits_pos = (z_q * z_k).sum(-1).unsqueeze(-1)
        logits_neg = torch.tensordot(z_q, queue, dims=1)

        # logits: Nx(1+K)
        logits = torch.cat([logits_pos, logits_neg], dim=-1)
        logits = logits.view(dims_prod, dims_prod)
        # apply temperature
        if self.T is not None:
            logits /= self.T

        label = torch.zeros(dims_prod, dtype=torch.long, device=z1.device)
        loss = self.criterion(logits, label)

        return loss
