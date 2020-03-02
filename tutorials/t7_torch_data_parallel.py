import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
from fastestimator.dataset import mnist
from fastestimator.op.numpyop import ExpandDims, Minmax
from fastestimator.pipeline import Pipeline


class LeNet(torch.nn.Module):
    def __init__(self, n_channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv2(x)
        x = fn.relu(x)
        x = fn.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, np.prod(x.size()[1:]))
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        x = fn.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    train_data, eval_data = mnist.load_data()
    device = torch.device("cuda:0")
    pipeline = Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=50,
                        ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    model = LeNet()
    model = nn.DataParallel(model)
    model.to(device)
    loader = pipeline.get_loader("train", 0)

    for data in loader:
        new_data = {}
        for key, val in data.items():
            new_data[key] = val.to(device)
        results = model(new_data["x"])
        pdb.set_trace()
