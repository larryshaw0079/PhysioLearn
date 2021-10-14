"""
@Time    : 2021/10/14 11:21
@File    : test_simclr_infonce_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch

from physiossl.learning import SimCLRInfoNCELoss


def test_forward():
    criterion = SimCLRInfoNCELoss()
    z1, z2 = torch.randn(32, 10, 20, 128), torch.randn(32, 10, 20, 128)
    loss = criterion(z1, z2)
    print(loss.item())
