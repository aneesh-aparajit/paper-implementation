import torch
import torch.nn as nn
import torch.optim as optim
import typing


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding))
        self.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(DenseBlock, self).__init__()
        self.add_module('1x1_conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_cha))

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
