"""
@Time    : 2021/6/23 16:46
@File    : deap.py
@Software: PyCharm
@Desc    : 
"""
import os
from typing import List

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from tqdm.std import tqdm
from torch.utils.data import Dataset

from .utils import standardize_tensor


class DEAPDataset(Dataset):
    num_subject = 32
    fs = 128

    def __init__(self, data_path: str, num_seq: int, subject_list: List, label_dim: int = 0, modal: str = 'eeg',
                 return_idx: bool = False, transform: nn.Module = None, standardize: str = 'none'):
        self.label_dim = label_dim
        self.return_idx = return_idx
        self.transform = transform

        assert modal in ['eeg', 'pps', 'all']

        # files = sorted(os.listdir(data_path))
        # assert len(files) == self.num_subject
        # files = [files[i] for i in subject_list]

        all_data = []
        all_labels = []

        for i, a_file in enumerate(subject_list):
            data = sio.loadmat(os.path.join(data_path, a_file))
            subject_data = data['data']  # trial x channel x data
            subject_label = data['labels']  # trial x label (valence, arousal, dominance, liking)
            # subject_data = tensor_standardize(subject_data, dim=-1)

            if modal == 'eeg':
                subject_data = subject_data[:, :32, :]
            elif modal == 'pps':
                subject_data = subject_data[:, 32:, :]
            elif modal == 'all':
                pass
            else:
                raise ValueError

            if standardize == 'none':
                pass
            elif standardize == 'standard':
                subject_data = standardize_tensor(subject_data, dim=-1)
            else:
                raise ValueError

            subject_data = subject_data.reshape(*subject_data.shape[:2], subject_data.shape[-1] // self.fs,
                                                self.fs)  # (trial, channel, num_sec, time_len)
            subject_data = np.swapaxes(subject_data, 1, 2)  # (trial, num_sec, channel, time_len)
            if num_seq == 0:
                subject_data = np.expand_dims(subject_data, axis=2)
            else:
                if subject_data.shape[1] % num_seq != 0:
                    subject_data = subject_data[:, :subject_data.shape[1] // num_seq * num_seq]
                subject_data = subject_data.reshape(subject_data.shape[0], subject_data.shape[1] // num_seq, num_seq,
                                                    *subject_data.shape[-2:])
                # (trial, *, *, channel, time_len)

            subject_label = np.repeat(np.expand_dims(subject_label, axis=1), subject_data.shape[1], axis=1)
            subject_label = np.repeat(np.expand_dims(subject_label, axis=2), subject_data.shape[2], axis=2)

            subject_data = subject_data.reshape(subject_data.shape[0] * subject_data.shape[1], *subject_data.shape[2:])
            subject_label = subject_label.reshape(subject_label.shape[0] * subject_label.shape[1],
                                                  *subject_label.shape[2:])

            all_data.append(subject_data)
            all_labels.append(subject_label)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if num_seq == 0:
            all_data = np.squeeze(all_data)

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
