"""test case for tensorflow users
"""
import os
import pdb

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


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
        x = F.softmax(x, dim=-1)
        return x

    def num_flat_features(self, x):
        return np.prod(x.size()[1:])


def get_estimator():
    #step 1
    pipeline = fe.Pipeline(train_data=get_dataloader(csv_path="/data/data/MNIST/train.csv", parrent_path="/data/data/MNIST/"),
                           eval_data=get_dataloader(csv_path="/data/data/MNIST/eval.csv", parrent_path="/data/data/MNIST/"))
    #step 2
    model = fe.build(model_def=Net, optimizer_def="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    #step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator
