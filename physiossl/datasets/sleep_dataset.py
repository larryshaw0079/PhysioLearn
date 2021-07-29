"""
@Time    : 2021/7/18 0:53
@File    : sleep_dataset.py
@Software: PyCharm
@Desc    : 
"""
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, root: str, url: str, download: bool = False):
        pass
