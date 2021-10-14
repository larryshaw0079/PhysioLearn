"""
@Time    : 2021/10/8 11:05
@File    : mlp.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feature_dim: int, num_class: int, hidden_dim: Union[int, List[int]] = None, num_layers: int = 2,
                 norm: bool = False, dropout: float = 0.0):
        super(MLP, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(feature_dim, num_class)
            )
        else:
            layers = []
            layers.append(nn.Linear(feature_dim, hidden_dim[0]))
            if norm:
                layers.append(nn.BatchNorm1d(hidden_dim[0]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            for i in range(1, len(hidden_dim)):
                layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                if norm:
                    layers.append(nn.BatchNorm1d(hidden_dim[i]))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim[-1], num_class))

    def forward(self, x):
        return self.layers(x)
