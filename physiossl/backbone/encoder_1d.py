"""
@Time    : 2021/6/23 17:08
@File    : encoder_1d.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, List

import torch
import torch.nn as nn


class R1DBlock(nn.Module):
    """
    The basic block of the 1d convolutional network

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, in_channel, out_channel, kernel_size=7, stride=1):
        super(R1DBlock, self).__init__()

        assert kernel_size % 2 == 1

        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel)
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layers(x)
        identity = self.downsample(x)

        out += identity

        return self.relu(out)


class R1DNet(nn.Module):
    def __init__(self, in_channel: int, mid_channel: int, feature_dim: int, layers: List = None,
                 kernel_size: Union[int, List[int]] = 7,
                 stride: Union[int, List[int]] = 1, final_fc: bool = True):
        super(R1DNet, self).__init__()

        self.final_fc = final_fc
        self.feature_size = mid_channel * 16

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 4
        else:
            raise ValueError

        if isinstance(stride, int):
            stride = [stride] * 4
        elif isinstance(stride, list):
            assert len(stride) == 4
        else:
            raise ValueError

        if layers is None:
            layers = [2, 2, 2, 2]

        self.head = nn.Sequential(
            nn.Conv1d(in_channel, mid_channel, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=7, stride=2, padding=3)
        )

        self.layer1 = self.__make_layer(layers[0], mid_channel, mid_channel * 2, kernel_size[0], stride[0])
        self.layer2 = self.__make_layer(layers[1], mid_channel * 2, mid_channel * 4, kernel_size[1], stride[1])
        self.layer3 = self.__make_layer(layers[2], mid_channel * 4, mid_channel * 8, kernel_size[2], stride[2])
        self.layer4 = self.__make_layer(layers[3], mid_channel * 8, mid_channel * 16, kernel_size[3], stride[3])

        if self.final_fc:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(mid_channel * 16, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0)

    def __make_layer(self, num_block, in_channel, out_channel, kernel_size, stride):
        layers = []

        layers.append(R1DBlock(in_channel, out_channel, kernel_size, stride))

        for _ in range(num_block):
            layers.append(R1DBlock(out_channel, out_channel, kernel_size, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.final_fc:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
