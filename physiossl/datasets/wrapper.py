"""
@Time    : 2021/10/5 2:01
@File    : wrapper.py
@Software: PyCharm
@Desc    : 
"""
import einops
import numpy as np
from torch.utils.data import Dataset
from statsmodels.tsa.stattools import adfuller


class CombinedTwoDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        assert len(dataset1) == len(dataset2)

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, item):
        return *self.dataset1[item], *self.dataset2[item]

    def __len__(self):
        return len(self.dataset1)


class RPDataset(Dataset):
    def __init__(self, src_dataset: Dataset):
        self.data = src_dataset.data
        self.labels = src_dataset.labels

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class TSDataset(Dataset):
    def __init__(self, src_dataset: Dataset):
        self.data = src_dataset.data
        self.labels = src_dataset.labels

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
