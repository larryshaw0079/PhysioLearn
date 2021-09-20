"""
@Time    : 2021/9/21 2:19
@File    : opportunity.py
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


class OpportunityUCIDataset(Dataset):
    def __init__(self, data_path: str, num_seq: int, subject_list: List = None, modal: str = 'eeg',
                 return_idx: bool = False,
                 transform: nn.Module = None, verbose: bool = True, standardize: str = 'none'):
        self.data_path = data_path
        self.transform = transform
        self.subject_list = subject_list
        self.modal = modal
        self.return_idx = return_idx
