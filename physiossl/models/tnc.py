"""
@Time    : 2021/6/23 16:57
@File    : tnc.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class TNC(nn.Module):
    def __init__(self, base_encoder: nn.Module, base_discriminator: nn.Module, feature_dim: int, weight: float = 0.05):
        super(TNC, self).__init__()

        self.encoder = base_encoder()
        self.discriminator = base_discriminator()
        self.w = weight

        self.criterion = nn.BCEWithLogitsLoss()
        self.neighbors = None
        self.non_neighbors = None

    def forward(self, x_t, x_p, x_n):
        z_t = self.encoder(x_t)
        z_p = self.encoder(x_p)
        z_n = self.encoder(x_n)

        d_p = self.discriminator(z_t, z_p)
        d_n = self.discriminator(z_t, z_n)

        if self.neighbors is None:
            self.neighbors = torch.ones((len(x_p))).cuda(x_t.device)
        if self.non_neighbors is None:
            self.non_neighbors = torch.zeros((len(x_n))).cuda(x_t.device)

        p_loss = self.criterion(d_p, self.neighbors)
        n_loss = self.criterion(d_n, self.non_neighbors)
        n_loss_u = self.criterion(d_n, self.neighbors)
        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

        return loss
