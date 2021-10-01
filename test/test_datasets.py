"""
@Time    : 2021/7/29 0:11
@File    : test_datasets.py
@Software: PyCharm
@Desc    : 
"""
import os

from physiossl.datasets import (
    AMIGOSDataset, BCICIV2aDataset, DEAPDataset, ISRUCDataset, OpportunityUCIDataset, SleepEDFDataset
)


# def test_amigos():
#     data_path = '/data/DataHub/EmotionRecognition/AMIGOS/signal'
#     files = list(filter(lambda x: x.endswith('mat'), os.listdir(data_path)))
#     dataset = AMIGOSDataset(data_path, 10, files, modal='all')
#
#     print(dataset[:10][0].shape, dataset[:10][1].shape)
#
#
# def test_deap():
#     data_path = '/data/DataHub/EmotionRecognition/DEAP/signal'
#     files = list(filter(lambda x: x.endswith('mat'), os.listdir(data_path)))
#     dataset = DEAPDataset(data_path, 10, files, modal='all')
#
#     print(dataset[:10][0].shape, dataset[:10][1].shape)
#
#
# def test_isruc():
#     data_path = '/data/DataHub/SleepClassification/isruc/isruc_full'
#     files = list(filter(lambda x: x.endswith('npz'), os.listdir(data_path)))
#     dataset = ISRUCDataset(data_path, 10, files, modal='all')
#
#     print(dataset[:10][0].shape, dataset[:10][1].shape)


def test_sleepedf():
    data_path = '/data/DataHub/SleepClassification/sleepedf153/sleepedf153'
    files = list(filter(lambda x: x.endswith('npz'), os.listdir(data_path)))
    dataset = SleepEDFDataset(data_path, 10, files, modal='all')

    print(dataset[:10][0].shape, dataset[:10][1].shape)
