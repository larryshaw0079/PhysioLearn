"""
@Time    : 2021/10/13 16:09
@File    : triplet_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, feature_dim, num_sample=10):
        super(TripletLoss, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.num_sample = num_sample
        self.device = device

    def forward(self, x1, x2):
        batch_size, *_ = x1.shape

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        pos = -F.logsigmoid((z1 * z2).sum(-1))
        neg_idx = torch.randint(low=0, high=batch_size, size=(batch_size, self.num_sample), device=x1.device)
        neg = torch.stack([z1[idx, :] for idx in neg_idx], dim=0)
        neg_sim = torch.einsum('mk,mjk->mj', [z1, neg]).sum(-1)

        assert pos.shape == neg_sim.shape, f'{pos.shape} - {neg_sim.shape}'

        loss = (pos + neg_sim).mean()

        return loss
