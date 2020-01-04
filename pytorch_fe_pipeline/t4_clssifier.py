import pdb
import time

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import fastestimator as fe
from fastestimator.op import TensorOp


class Normalize(TensorOp):
    def forward(self, data, state):
        data = (data - 0.5) / 0.5
        return data


def get_in_memory_pipeline(batch_size):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    train_data = {"x": np.float32(np.expand_dims(x_train, -1) / 255), "y": np.int64(y_train)}
    eval_data = {"x": np.float32(np.expand_dims(x_eval, -1) / 255), "y": np.int64(y_eval)}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Normalize(inputs="x", outputs="x"))
    return pipeline


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

    # pipeline = get_in_memory_pipeline(32)
    # pipeline.global_batch_multiplier = 1
    # pipeline.prepare()
    # ds_iter = pipeline.dataset_schedule["train"].get_current_value(0)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5], inplace=True)])

    trainset = torchvision.datasets.MNIST(root="/data/data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    testnset = torchvision.datasets.MNIST(root="/data/data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testnset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    #get some random training images
    dataiter = iter(trainloader)
    # tic = time.perf_counter()
    # for idx in range(1850):
    #     images, labels = next(dataiter)
    #     if idx > 0 and idx % 100 == 0:
    #         elapse = time.perf_counter() - tic
    #         example_sec = 100 * 32 / elapse
    #         print("step: {}, image/sec: {}".format(idx, example_sec))
    #         tic = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    # net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    tic = time.perf_counter()
    running_loss = 0.0
    for i in range(10000):
        # batch_data = next(ds_iter)
        # x = batch_data["x"]
        # y = batch_data["y"]

        # x = torch.tensor(batch_data["x"].numpy())
        # y = torch.tensor(batch_data["y"].numpy())

        x, y = next(dataiter)
        pdb.set_trace()
        # x = x.to(device)
        # y = y.to(device)
        # x = x.permute(0, 3, 1, 2)
        # #zero the parameter gradients
        optimizer.zero_grad()

        # #forward+backward+optimize
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.to("cpu").item()
        # print(loss.to("cpu"))
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%5d] loss: %.3f' % (i + 1, running_loss / 100))
            running_loss = 0.0
            elapse = time.perf_counter() - tic
            example_sec = 100 * 32 / elapse
            print("step: {}, image/sec: {}".format(i, example_sec))
            tic = time.perf_counter()
