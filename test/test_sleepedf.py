"""
@Time    : 2021/7/29 0:11
@File    : test_sleepedf.py
@Software: PyCharm
@Desc    : 
"""
from physiossl.datasets import SleepEDF39


def test_sleep_edf39():
    dataset = SleepEDF39('/data/DataHub/SleepStageClassification/sleepedf39', download=True)
