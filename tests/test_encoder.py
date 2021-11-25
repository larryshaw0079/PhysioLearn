"""
@Time    : 2021/10/13 15:23
@File    : test_encoder.py
@Software: PyCharm
@Desc    : 
"""
from torchsummary import summary

from physiossl.backbone import resnet_1d, convnet_1d


def test_resnet_1d():
    model = resnet_1d(2, 5)
    summary(model, input_size=(2, 3000), device='cpu')
    num_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'[INFO] Total number of parameters: {num_params}...')


def test_convnet_1d():
    model = convnet_1d(2, 5)
    summary(model, input_size=(2, 3000), device='cpu')
    num_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'[INFO] Total number of parameters: {num_params}...')
