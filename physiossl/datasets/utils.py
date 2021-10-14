"""
@Time    : 2021/7/18 0:58
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from einops import reduce

EPS = 1e-6


def standard_scale(x: np.ndarray, dim=-1):
    x_mean = np.mean(x, axis=dim, keepdims=True)
    x_std = np.std(x, axis=dim, keepdims=True)

    return (x - x_mean) / (x_std + EPS)


def minmax_scale(x: np.ndarray, dim=-1):
    x_min = np.min(x, axis=dim, keepdims=True)
    x_max = np.max(x, axis=dim, keepdims=True)

    return (x - x_min) / (x_max - x_min + EPS)
