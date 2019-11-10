import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5], inplace=True)])
    trainset = torchvision.datasets.MNIST(root="/data/data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

    testnset = torchvision.datasets.MNIST(root="/data/data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testnset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

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
    device = torch.device("cuda")
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    tic = time.perf_counter()
    i = 0
    for epoch in range(1):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)

            # #zero the parameter gradients
            optimizer.zero_grad()

            # #forward+backward+optimize
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%5d] loss: %.3f' % (i + 1, loss.to("cpu").item()))
                # print(loss)
                running_loss = 0.0
                elapse = time.perf_counter() - tic
                example_sec = 100 * 32 / elapse
                print("step: {}, image/sec: {}".format(i, example_sec))
                tic = time.perf_counter()