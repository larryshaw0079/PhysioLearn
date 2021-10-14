"""
@Time    : 2021/10/9 10:58
@File    : simclr.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
