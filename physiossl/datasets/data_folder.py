"""
@Time    : 2021/11/25 15:31
@File    : data_folder.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, Tuple, Iterable

import torch.nn as nn
from torch.utils.data import Dataset


def read_physio_data(root: str) -> Dataset:
    return DataFolder(root)


class DataFolder(Dataset):
    def __init__(self, root: str, suffix: str = None, file_list: Iterable[str] = None, transform: nn.Module = None,
                 channel_range: Union[Tuple[int, int], Iterable[str]] = None, label_name: str = None,
                 label_dim: int = 0, standardization: str = 'none', return_idx: bool = False):
        pass

    @property
    def num_subjects(self):
        pass
