"""
@Time    : 2021/7/18 0:58
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def standardize_tensor(x: np.ndarray, dim=-1):
    x_min = np.expand_dims(x.min(axis=dim), axis=dim)
    x_max = np.expand_dims(x.max(axis=dim), axis=dim)

    return (x - x_min) / (x_max - x_min)


class BiDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, item):
        return *(self.dataset1[item]), *(self.dataset2[item])

    def __len__(self):
        return len(self.dataset1)
