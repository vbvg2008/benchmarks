"""test case for tensorflow users
"""
import os
import pdb
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MnistDataset(Dataset):
    def __init__(self, csv_path, parrent_path):
        self.csv_path = csv_path
        self.parrent_path = parrent_path
        self.data = dict()
        self.read_csv()

    def read_csv(self):
        df = pd.read_csv(self.csv_path)
        self.data = df.to_dict('list')

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        data = {}
        for key in self.data:
            data[key] = self.data[key][idx]
        return self.transform_data(data)

    def transform_data(self, data):
        data["x"] = cv2.imread(os.path.join(self.parrent_path, data["x"]), cv2.IMREAD_GRAYSCALE)
        data["x"] = np.expand_dims(data["x"], 0)
        data["x"] = data["x"] / 255.0
        data["x"] = np.float32(data["x"])
        return data


def get_dataloader(csv_path, parrent_path):
    dataset = MnistDataset(csv_path=csv_path, parrent_path=parrent_path)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=8)
    return loader


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        return np.prod(x.size()[1:])


if __name__ == "__main__":
    train_loader = get_dataloader(csv_path="/data/data/MNIST/train.csv", parrent_path="/data/data/MNIST/")
    eval_loader = get_dataloader(csv_path="/data/data/MNIST/eval.csv", parrent_path="/data/data/MNIST/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    # net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.zero_grad()
    tic = time.perf_counter()
    for idx, batch in enumerate(train_loader):
        x = batch["x"]
        y = batch["y"]

        # x = x.to(device)
        # y = y.to(device)

        # optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # running_loss += loss.to("cpu").item()
        if idx % 100 == 0:  # print every 100 mini-batches
            print('loss: %.3f' % loss.item())
            # print('loss: %.3f' % loss.to("cpu").item())
            elapse = time.perf_counter() - tic
            example_sec = 100 * 32 / elapse
            print("step: {}, image/sec: {}".format(idx, example_sec))
            tic = time.perf_counter()
