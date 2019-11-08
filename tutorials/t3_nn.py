import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


net = Net()
#print the architecture
print(net)

params = list(net.parameters())

# inference an image
x = torch.randn(1, 1, 28, 28)
out = net(x)
print(out.size())
loss_obj = nn.CrossEntropyLoss()
loss = loss_obj(out, torch.tensor([2]))

print(loss)

# zero out gradient
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

optimizer = torch.optim.SGD(net.parameters(), lr=100000)
optimizer.zero_grad()

print(net.conv1.weight.data)
x = torch.randn(1, 1, 28, 28)
out = net(x)
loss_obj = nn.CrossEntropyLoss()
loss = loss_obj(out, torch.tensor([2]))
loss.backward()
optimizer.step()
print(net.conv1.bias.data)
