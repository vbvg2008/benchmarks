import pdb
import time

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
        # print(x.shape)
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
        # x = fn.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    train_data, eval_data = mnist.load_data()
    device = torch.device("cuda:0")
    pipeline = Pipeline(train_data=train_data,
                        eval_data=eval_data,
                        batch_size=32,
                        ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
    # pdb.set_trace()
    model = nn.DataParallel(model)
    model.to(device)
    loader = pipeline.get_loader("train", 0)
    criterion = nn.CrossEntropyLoss()
    tic = time.perf_counter()
    i = 0
    for _ in range(5):
        for data in loader:
            x = data["x"].to(device)
            y = data["y"].to(device)
            # #zero the parameter gradients
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%5d] loss: %.3f' % (i + 1, loss.to("cpu").item()))
                # print(loss)
                running_loss = 0.0
                elapse = time.perf_counter() - tic
                example_sec = 100 / elapse
                print("step: {}, steps/sec: {}".format(i, example_sec))
                tic = time.perf_counter()
