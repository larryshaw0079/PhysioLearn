"""
@Time    : 2021/6/23 16:46
@File    : sleepedf.py
@Software: PyCharm
@Desc    : 
"""
import os
from pathlib import Path
from typing import Union

from .sleep_dataset import SleepDataset
from .utils import download_url, extract_archive


class SleepEDF39(SleepDataset):
    """
    SleepEDF39 dataset

    Attributes
    ----------

    Methods
    -------
    """
    url = 'https://physionet.org/static/published-projects/sleep-edf/sleep-edf-database-1.0.0.zip'
    meta = {
        'md5': 'd499251742e5b7e044820576268a481e'
    }

    def __init__(self, root: Union[str, Path], url: str = None, download: bool = False):
        if url is None:
            url = self.url

        if download:
            if self.__check_integrity():
                print('Files already downloaded and verified')
            else:
                download_url(url, os.path.join(root, 'raw'), md5=self.meta['md5'])
                extract_archive(os.path.join(root, 'raw', 'sleep-edf-database-1.0.0.zip'))

    def __check_integrity(self):
        pass

    def __preprocess(self):
        pass


class SleepEDF153(SleepDataset):
    """"""
    url = 'https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip'
    md5 = '30b79d7c3607e6d021d0c16ff346842a'

    def __init__(self, download: bool = False):
        pass

    def __preprocess(self):
        pass
