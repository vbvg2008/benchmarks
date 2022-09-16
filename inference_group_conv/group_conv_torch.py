from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

from util import timeit


class ResNet9(nn.Module):
    """A 9-layer ResNet PyTorch model for cifar10 image classification.
    The model architecture is from https://github.com/davidcpage/cifar10-fast
    Args:
        input_size: The size of the input tensor (channels, height, width). Both width and height of input_size should
            not be smaller than 16.
        classes: The number of outputs.
    Raises:
        ValueError: Length of `input_size` is not 3.
        ValueError: `input_size`[1] or `input_size`[2] is not a multiple of 16.
    """
    def __init__(self, input_size: Tuple[int, int, int] = (3, 32, 32), classes: int = 10):
        super().__init__()
        self.conv0 = nn.Conv2d(input_size[0], 128, 3, padding=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(128, momentum=0.2)
        self.conv1 = nn.Conv2d(128, 256, 3, padding=(1, 1), groups=2)
        self.conv1_bn = nn.BatchNorm2d(256, momentum=0.2)
        self.residual1 = Residual(256)
        self.conv2 = nn.Conv2d(256, 512, 3, padding=(1, 1), groups=2)
        self.conv2_bn = nn.BatchNorm2d(512, momentum=0.2)
        self.residual2 = Residual(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=(1, 1), groups=2)
        self.conv3_bn = nn.BatchNorm2d(1024, momentum=0.2)
        self.residual3 = Residual(1024)
        self.fc1 = nn.Linear(1024, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prep layer
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = fn.softmax(x, dim=-1)
        return x


class Residual(nn.Module):
    """A two-layer unit for ResNet9. The output size is the same as input.
    Args:
        channel: Number of input channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, 1), groups=2)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, 1), groups=2)
        self.conv2_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = fn.leaky_relu(x, negative_slope=0.1)
        return x


def single_inference(model, data):
    output = model(data)
    return output


if __name__ == "__main__":
    model = ResNet9(input_size=(3, 1024, 1024), classes=100)
    data = torch.Tensor(np.random.rand(1, 3, 1024, 1024).astype("float32"))
    model.to("cuda")
    data = data.to("cuda")
    timeit(f=lambda: single_inference(model=model, data=data), num_runs=500)
