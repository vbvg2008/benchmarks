import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from matplotlib import pyplot as plt

model_dir = "/tmp/tmpq6wsmr1e/model_epoch_45.pt"


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 7 * 7 * 256, bias=False)
        self.bn1d = nn.BatchNorm1d(7 * 7 * 256)
        self.conv_tran1 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(128)
        self.conv_tran2 = nn.ConvTranspose2d(128, 64, 5, stride=2, bias=False, padding=2, output_padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv_tran3 = nn.ConvTranspose2d(64, 1, 5, stride=2, bias=False, padding=2, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1d(x)
        x = fn.leaky_relu(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_tran1(x)
        x = self.bn2d1(x)
        x = fn.leaky_relu(x)
        x = self.conv_tran2(x)
        x = self.bn2d2(x)
        x = fn.leaky_relu(x)
        x = self.conv_tran3(x)
        x = torch.tanh(x)
        return x


def plot_and_save(images):
    fig, axes = plt.subplots(4, 4)
    axes = np.ravel(axes)
    for i in range(images.shape[0]):
        axes[i].axis('off')
        axes[i].imshow(np.squeeze(images[i, ...] * 127.5 + 127.5), cmap='gray')
    plt.savefig("sample.png")


if __name__ == "__main__":
    gen_model = Generator()
    gen_model.load_state_dict(torch.load(model_dir))
    gen_model.eval()
    vector = torch.randn(16, 100)
    images = gen_model(vector).detach().numpy()
    plot_and_save(images)
