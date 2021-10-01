"""
@Time    : 2021/9/21 2:08
@File    : bciciv2.py
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


class BCICIV2aDataset(Dataset):
    def __init__(self, data_path: str, num_seq: int, subject_list: List = None, modal: str = 'eeg',
                 return_idx: bool = False,
                 transform: nn.Module = None, verbose: bool = True, standardize: str = 'none'):
        assert isinstance(subject_list, list)

        self.data_path = data_path
        self.transform = transform
        self.subject_list = subject_list
        self.modal = modal
        self.return_idx = return_idx

        event_types = [769, 770, 771, 772, 783]

        assert modal in ['eeg', 'pps', 'all']

        self.data = []
        self.labels = []

        for i, patient in enumerate(tqdm(subject_list, desc='::: LOADING DATA ::::')):
            print(f'[INFO] Processing patient {patient}...')

            data = np.load(os.path.join(data_path, patient))
            raw = data['s']
            types = data['etyp']
            pos = data['epos'].flatten()
            durations = data['edur'].flatten()

            event_idx = np.array([i for i in range(len(types)) if types[i] in event_types])
            event_pos = pos[event_idx]
            event_durations = durations[event_idx]

            assert (event_durations == event_durations[0]).all()

            raw = raw.T
            raw_segments = [raw[:, p: p + event_durations[idx]] for idx, p in enumerate(event_pos)]
            raw_segments = np.stack(raw_segments, axis=0)

            if modal == 'eeg':
                recordings = raw_segments[:, :22]
            elif modal == 'pps':
                recordings = raw_segments[:, 22:]
            elif modal == 'all':
                recordings = raw_segments
            else:
                raise ValueError

            label = sio.loadmat(os.path.join(data_path, '2a_label', patient.split('.')[0] + '.mat'))
            annotations = label['classlabel'].flatten() - 1

            recordings = recordings[:(recordings.shape[0] // num_seq) * num_seq].reshape(-1, num_seq,
                                                                                         *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_seq) * num_seq].reshape(-1, num_seq)

            assert recordings.shape[:2] == annotations.shape[:2], f'{patient}: {recordings.shape} - {annotations.shape}'

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = x.astype(np.float32)
        y = y.astype(np.long)

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
