import os
import pdb

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.keras import Sequential, layers
from torch.utils.data import DataLoader, Dataset

import fastestimator as fe


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


def torch_model():
    return Net()


def tf_model(input_shape=(28, 28, 1), classes=10):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


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
        data["x"] = np.expand_dims(data["x"], -1)
        data["x"] = data["x"] / 255.0
        return data


def get_estimator():
    dataset_train = MnistDataset(csv_path="/data/data/MNIST/train.csv", parrent_path="/data/data/MNIST/")
    dataset_eval = MnistDataset(csv_path="/data/data/MNIST/eval.csv", parrent_path="/data/data/MNIST/")
    train_loader = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=dataset_eval, batch_size=32, shuffle=True, num_workers=8)

    model = fe.build(model_def=torch_model, optimizer_def="sgd")
    pdb.set_trace()
    return model


if __name__ == "__main__":
    model = get_estimator()
#     ds_iter = iter(dataloader)
#     tic = time.perf_counter()
#     for idx in range(1850):
#         batch_data = next(ds_iter)
#         if idx > 0 and idx % 100 == 0:
#             elapse = time.perf_counter() - tic
#             example_sec = 100 * 32 / elapse
#             print("step: {}, image/sec: {}".format(idx, example_sec))
#             tic = time.perf_counter()

torch.nn.CrossEntropyLoss()

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

