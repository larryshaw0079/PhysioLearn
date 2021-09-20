"""
@Time    : 2021/7/18 0:58
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np


def standardize_tensor(x: np.ndarray, dim=-1):
    x_min = np.expand_dims(x.min(axis=dim), axis=dim)
    x_max = np.expand_dims(x.max(axis=dim), axis=dim)

    return (x - x_min) / (x_max - x_min)
