"""
@Time    : 2021/6/23 16:56
@File    : rp.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class RelativePosition(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, feature_dim, device='cuda'):
        super(RelativePosition, self).__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.device = device

        self.encoder = R1DNetSimple(input_size, input_channels, feature_dim)
        # self.encoder = R1DNet(input_channels, mid_channel=16, feature_dim=feature_dim, stride=2,
        #                       kernel_size=[7, 11, 11, 7],
        #                       final_fc=True)

        self.linear_head = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, x1, x2):
        # print('---------X1', x1.shape)
        # print('---------X2', x2.shape)

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # print('---------Z1', z1.shape)
        # print('---------Z2', z2.shape)

        out = torch.abs(z1 - z2)
        out = self.linear_head(out)

        return out
