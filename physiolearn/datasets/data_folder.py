"""
@Time    : 2021/11/25 15:31
@File    : data_folder.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
from pathlib import Path
from typing import Union, Tuple, Iterable

import numpy as np
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Dataset


def read_physio_data(root: str) -> Dataset:
    return DataFolder(root)


class DataFolder(Dataset):
    def __init__(self, root: str, suffix: str = None, file_list: Iterable[str] = None, transform: nn.Module = None,
                 target_transform: nn.Module = None, data_attributes: Union[Iterable[str], str] = None,
                 label_attribute: str = None, channel_last: bool = False,
                 channel_range: Tuple[int, int] = None, standardization: str = 'none'):
        assert (suffix is not None) ^ (file_list is not None)
        if suffix is not None:
            file_list = Path(root).glob(f'*{suffix}')

        data = []
        targets = []

        for file_name in file_list:
            file_path = os.path.join(root, file_name.name if isinstance(file_name, Path) else file_name)
            if file_path.endswith('.npz'):
                data_dict = np.load(file_path)
            elif file_path.endswith('.mat'):
                data_dict = sio.loadmat(file_path)
            else:
                raise ValueError('Unsupported data type!')

            if data_attributes is None:
                assert label_attribute is not None
                warnings.warn(
                    '`data_attributes` is not specified, using all attributes except `label_attribute` as default.')
                data_attributes = list(
                    filter(lambda item: item != label_attribute and not item.startswith('__'), data_dict.keys()))

            if isinstance(data_attributes, str):
                data_attributes = [data_attributes]

            data_subject = []
            for attribute in data_attributes:
                if channel_range is not None:
                    if channel_last:
                        data_current = data_dict[attribute][..., channel_range[0]: channel_range[1]]
                        data_current = np.swapaxes(data_current, -1, -2)
                    else:
                        data_current = data_dict[attribute][..., channel_range[0]: channel_range[1], :]
                else:
                    data_current = data_dict[attribute]

                data_subject.append(data_current)
            data_subject = np.stack(data_subject)

    @property
    def num_subjects(self):
        pass
