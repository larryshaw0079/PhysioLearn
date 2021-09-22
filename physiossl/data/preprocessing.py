"""
@Time    : 2021/9/21 2:22
@File    : preprocessing.py
@Software: PyCharm
@Desc    : 
"""
import os
import numpy as np

from sklearn.model_selection import KFold, LeaveOneOut


def cross_validation_splitting(data_path: str, kfold: int = 10, suffix: str = ''):
    files = list(filter(lambda x: x.endswith(suffix), os.listdir(data_path)))
    files = np.sort(files)

    assert len(
        files) >= kfold, f'Insufficient number of subjects to perform {kfold}-fold cross validation. Try leave one out instread.'
    kf = KFold(n_splits=kfold)
    for i, (train_index, test_index) in enumerate(kf.split(files)):
        train_subjects, test_subjects = files[train_index].tolist(), files[test_index].tolist()
        yield train_subjects, test_subjects


def leave_one_out_splitting(data_path: str, suffix: str = ''):
    files = list(filter(lambda x: x.endswith(suffix), os.listdir(data_path)))
    files = np.sort(files)

    loo = LeaveOneOut()
    print('[INFO] Generated', loo.get_n_splits(files), 'splits...')
    for i, (train_index, test_index) in enumerate(loo.split(files)):
        train_subjects, test_subjects = files[train_index].tolist(), files[test_index].tolist()
        yield train_subjects, test_subjects
