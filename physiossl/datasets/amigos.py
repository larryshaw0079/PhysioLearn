"""
@Time    : 2021/6/23 16:46
@File    : amigos.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
from typing import List

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from tqdm.std import tqdm
from torch.utils.data import Dataset

from .utils import standardize_tensor


class AMIGOSDataset(Dataset):
    num_subject = 40
    fs = 128

    def __init__(self, data_path: str, num_seq: int, subject_list: List, label_dim: int = 0, modal: str = 'eeg',
                 return_idx: bool = False, transform: nn.Module = None, standardize: str = 'none'):
        self.transform = transform
        self.label_dim = label_dim
        self.return_idx = return_idx

        assert modal in ['eeg', 'pps', 'all']

        all_data = []
        all_labels = []

        for i, a_file in enumerate(tqdm(subject_list, desc='::: LOADING DATA ::::')):
            data = sio.loadmat(os.path.join(data_path, a_file))

            subject_data = []
            subject_label = []
            for i in range(data['joined_data'].shape[1]):
                trial_data = data['joined_data'][0, i]
                trial_label = data['labels_selfassessment'][0, i]

                if standardize == 'none':
                    pass
                elif standardize == 'standard':
                    trial_data = standardize_tensor(trial_data, dim=0)
                else:
                    raise ValueError

                trial_data = trial_data[:trial_data.shape[0] // self.fs * self.fs]
                trial_data = trial_data.reshape(trial_data.shape[0] // self.fs, self.fs,
                                                trial_data.shape[-1])
                trial_data = np.swapaxes(trial_data, 1, 2)

                if np.isnan(trial_data).any():
                    warnings.warn(
                        f"The array of {a_file} - {i} contains {np.sum(np.isnan(trial_data))} NaN of total {np.prod(trial_data.shape)} points, dropped.")
                    continue

                if modal == 'eeg':
                    trial_data = trial_data[:, :14]
                elif modal == 'pps':
                    trial_data = trial_data[:, 14:]
                elif modal == 'all':
                    pass
                else:
                    raise ValueError

                if trial_data.shape[0] % num_seq != 0:
                    trial_data = trial_data[:trial_data.shape[0] // num_seq * num_seq]

                # Standardize
                # mean_value = np.expand_dims(trial_data.mean(axis=0), axis=0)
                # std_value = np.expand_dims(trial_data.std(axis=0), axis=0)
                # trial_data = (trial_data - mean_value) / std_value

                trial_data = trial_data.reshape(trial_data.shape[0] // num_seq, num_seq, *trial_data.shape[1:])

                if 0 in trial_data.shape:
                    warnings.warn(f"The array of shape {data['joined_data'][0, i].shape} is too small, dropped.")
                    continue

                trial_label = np.repeat(trial_label, trial_data.shape[1], axis=0)
                trial_label = np.repeat(np.expand_dims(trial_label, axis=0), trial_data.shape[0], axis=0)

                if 0 in trial_label.shape:
                    warnings.warn(f"The label of {a_file} - {i} is malfunctioned, dropped.")
                    continue

                subject_data.append(trial_data)
                subject_label.append(trial_label)

            subject_data = np.concatenate(subject_data, axis=0)
            subject_label = np.concatenate(subject_label, axis=0)

            print(subject_data.shape, subject_label.shape)

            all_data.append(subject_data)
            all_labels.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        self.data = all_data
        self.labels = all_labels
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item].astype(np.float32)
        label = self.labels[item].astype(np.long)[:, self.label_dim]
        y = np.zeros_like(label, dtype=np.long)
        y[label >= 5] = 1

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)
