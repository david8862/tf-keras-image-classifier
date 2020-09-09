#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=1, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_bn_relu6 = nn.Sequential(
             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False),
             nn.BatchNorm2d(16),
             nn.ReLU6(inplace=True))

        self.sepconv_bn_relu6 = nn.Sequential(
             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False),
             nn.BatchNorm2d(16),
             nn.ReLU6(inplace=True))

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





#if __name__ == "__main__":
    #input_shape = (64, 64, 1)
    #num_classes = 7
    #model = tiny_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = mini_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = big_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = simple_CNN((48, 48, 1), num_classes)
    #model.summary()
