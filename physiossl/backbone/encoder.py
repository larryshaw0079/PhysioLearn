"""
@Time    : 2021/6/23 17:08
@File    : encoder.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, List

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    The basic block of the 1d residual convolutional network
    """

    def __init__(self, in_channel, out_channel, kernel_size=7, stride=1):
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


# class ResNet1D(nn.Module):
#     def __init__(self, in_channel: int, mid_channel: int, feature_dim: int, layers: List = None,
#                  kernel_size: Union[int, List[int]] = 7,
#                  stride: Union[int, List[int]] = 1, final_fc: bool = True):
#         super(ResNet1D, self).__init__()
#
#         self.final_fc = final_fc
#         self.feature_size = mid_channel * 16
#
#         if isinstance(kernel_size, int):
#             kernel_size = [kernel_size] * 4
#         elif isinstance(kernel_size, list):
#             assert len(kernel_size) == 4
#         else:
#             raise ValueError
#
#         if isinstance(stride, int):
#             stride = [stride] * 4
#         elif isinstance(stride, list):
#             assert len(stride) == 4
#         else:
#             raise ValueError
#
#         if layers is None:
#             layers = [2, 2, 2, 2]
#
#         self.head = nn.Sequential(
#             nn.Conv1d(in_channel, mid_channel, kernel_size=7, stride=2,
#                       padding=3, bias=False),
#             nn.BatchNorm1d(mid_channel),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=7, stride=2, padding=3)
#         )
#
#         self.layer1 = self.__make_layer(layers[0], mid_channel, mid_channel * 2, kernel_size[0], stride[0])
#         self.layer2 = self.__make_layer(layers[1], mid_channel * 2, mid_channel * 4, kernel_size[1], stride[1])
#         self.layer3 = self.__make_layer(layers[2], mid_channel * 4, mid_channel * 8, kernel_size[2], stride[2])
#         self.layer4 = self.__make_layer(layers[3], mid_channel * 8, mid_channel * 16, kernel_size[3], stride[3])
#
#         if self.final_fc:
#             self.avgpool = nn.AdaptiveAvgPool1d(1)
#             self.fc = nn.Linear(mid_channel * 16, feature_dim)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0.0)
#
#     def __make_layer(self, num_block, in_channel, out_channel, kernel_size, stride):
#         layers = []
#
#         layers.append(ResidualBlock1D(in_channel, out_channel, kernel_size, stride))
#
#         for _ in range(num_block):
#             layers.append(ResidualBlock1D(out_channel, out_channel, kernel_size, 1))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.head(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.final_fc:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
#
# class SimpleConvNet1D(nn.Module):
#     def __init__(self, in_channel: int, feature_dim: int, kernel_size: Union[int, List[int]],
#                  strides: Union[int, List[int]], dropout: float):
#         super(SimpleConvNet1D, self).__init__()
#
#         if isinstance(kernel_size, int):
#             kernel_size = [kernel_size] * 4
#
#         if isinstance(strides, int):
#             strides = [strides] * 4
#
#         assert len(kernel_size) == len(strides) == 4
#
#         self.feature_dim = feature_dim
#
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(in_channel, 32, kernel_size=kernel_size[0],
#                       stride=strides[0], bias=False, padding=kernel_size[0] // 2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(dropout)
#         )
#
#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=kernel_size[1], stride=strides[1], bias=False, padding=kernel_size[1] // 2),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )
#
#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, 128, kernel_size=kernel_size[2], stride=strides[2], bias=False, padding=kernel_size[2] // 2),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )
#
#         self.conv_block4 = nn.Sequential(
#             nn.Conv1d(128, 256, kernel_size=kernel_size[3], stride=strides[3], bias=False, padding=kernel_size[3] // 2),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )
#
#         self.avg = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, feature_dim)
#
#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.conv_block4(x)
#
#         x = self.avg(x)
#         x = x.squeeze()
#
#         return self.fc(x)


class ConvNet1D(nn.Module):
    def __init__(self, basic_block: nn.Module, in_channel: int, hidden_channel: int, kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]], num_layers: List[int], classes: int):
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
    return ConvNet1D(ResidualBlock1D, in_channel=in_channel, hidden_channel=16, kernel_size=[7, 11, 11, 7],
                     stride=[1, 2, 2, 2], num_layers=[2, 2, 2, 2], classes=classes)


def convnet_1d(in_channel: int, classes: int):
    return ConvNet1D(BasicConvBlock1D, in_channel=in_channel, hidden_channel=16, kernel_size=[7, 11, 11, 7],
                     stride=[1, 2, 2, 2], num_layers=[2, 2, 2, 2], classes=classes)
