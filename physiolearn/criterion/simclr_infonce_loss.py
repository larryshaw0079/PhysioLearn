"""
@Time    : 2021/10/13 16:07
@File    : simclr_infonce_loss.py
@Software: PyCharm
@Desc    : 
"""
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physiolearn.dist.utils import gather_tensor_sync, is_distributed_enabled


class SimCLRInfoNCELoss(nn.Module):
    def __init__(self, T: float = 1.0):
        super(SimCLRInfoNCELoss, self).__init__()

        self.T = T
        self.criterion = nn.CrossEntropyLoss()
        self.mask = None

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """

        Args:
            z1 (torch.Tensor): with shape (B, *, F)
            z2 (torch.Tensor): with shape (B, *, F)

        Returns:

        """
        assert z1.shape == z2.shape

        if is_distributed_enabled():
            z1 = gather_tensor_sync(z1)
            z2 = gather_tensor_sync(z2)

        B, *_ = z1.shape  ## (batch, *, feature_dim)
        feature_dims = z1.shape[:-1]  ## without the representation dim
        dims_prod = np.prod(feature_dims)

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        pos_logits = (z1 * z2).sum(-1).unsqueeze(-1)
        # neg_logits = torch.einsum('ijk,mnk->ijnm', [z1, z2])
        neg_logits = torch.tensordot(z1, z2.T, dims=1)

        if self.mask is None:
            # identifiers = torch.arange(np.prod(z1.shape[:-1]), dtype=torch.long, device=z1.device).view(*z1.shape[:-1]).unsqueeze(-1)
            # mask = torch.eq(identifiers, identifiers.T)
            mask = torch.ones(*feature_dims, *feature_dims[::-1], dtype=torch.bool, device=z1.device)
            for idx_tuple in itertools.product(*(range(s) for s in feature_dims)):
                mask[idx_tuple + idx_tuple[::-1]] = False
            self.mask = mask

        neg_logits = neg_logits.masked_select(self.mask).view(*feature_dims, dims_prod - 1)  ## mask out selves
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits.view(dims_prod, dims_prod)
        if self.T is not None:
            logits /= self.T

        label = torch.zeros(dims_prod, dtype=torch.long, device=z1.device)
        loss = self.criterion(logits, label)

        return loss
