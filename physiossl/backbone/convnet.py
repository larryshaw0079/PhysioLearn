"""
@Time    : 2021/6/23 17:08
@File    : convnet.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, List

import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    The basic block of the 1d residual convolutional network
    """

    def __init__(self, in_channel, out_channel, kernel_size=7, stride=1):
        """

        Args:
            in_channel ():
            out_channel ():
            kernel_size ():
            stride ():
        """
        super(ResidualBlock1D, self).__init__()

        # assert kernel_size % 2 == 1

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


class BasicConvBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, stride=1):
        """

        Args:
            in_channel ():
            out_channel ():
            kernel_size ():
            stride ():
        """
        super(BasicConvBlock1D, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class ConvNet1D(nn.Module):
    def __init__(self, basic_block: nn.Module, in_channel: int, hidden_channel: int, kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]], num_layers: List[int], classes: int):
        """

        Args:
            basic_block ():
            in_channel ():
            hidden_channel ():
            kernel_size ():
            stride ():
            num_layers ():
            classes ():
        """
        super(ConvNet1D, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(num_layers)
        if isinstance(stride, int):
            stride = [stride] * len(num_layers)

        assert len(kernel_size) == len(stride) == len(num_layers)

        self.head = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_channel = hidden_channel

        conv_layers = []
        for i, nl in enumerate(num_layers):
            conv_layers.append(self.__make_layer(basic_block, nl, self.in_channel * 2, kernel_size[i], stride[i]))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.in_channel, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, num_blocks, out_channel, kernel_size, stride):
        layers = []
        layers.append(block(self.in_channel, out_channel, kernel_size, stride))
        self.in_channel = out_channel

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel, kernel_size, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.head(x)
        out = self.conv_layers(out)
        out = self.avg_pool(out)
        out = out.squeeze()
        out = self.fc(out)

        return out


def resnet_1d(in_channel: int, classes: int):
    """

    Args:
        in_channel ():
        classes ():

    Returns:

    """
    return ConvNet1D(ResidualBlock1D, in_channel=in_channel, hidden_channel=16, kernel_size=[7, 11, 11, 7],
                     stride=[1, 2, 2, 2], num_layers=[2, 2, 2, 2], classes=classes)


def convnet_1d(in_channel: int, classes: int):
    """

    Args:
        in_channel ():
        classes ():

    Returns:

    """
    return ConvNet1D(BasicConvBlock1D, in_channel=in_channel, hidden_channel=16, kernel_size=[7, 11, 11, 7],
                     stride=[1, 2, 2, 2], num_layers=[2, 2, 2, 2], classes=classes)
