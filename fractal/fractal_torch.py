import pdb
import tempfile

import fastestimator as fe
import torch
import torch.nn as nn
import torch.nn.functional as fn
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, padding=k // 2)
        self.batch_norm = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class Res2NetBlock(nn.Module):
    def __init__(self, c1, c2, s=4):
        super().__init__()
        cg = c2 // s
        self.conv_in = ConvBlock(c1=c1, c2=c2, k=1)
        self.conv_m1 = ConvBlock(c1=cg, c2=cg, k=3)
        self.conv_m2 = ConvBlock(c1=cg, c2=cg, k=3)
        self.conv_m3 = ConvBlock(c1=cg, c2=cg, k=3)
        self.conv_out = ConvBlock(c1=c2, c2=c2, k=1)
        self.add = c1 == c2
        self.cg = cg

    def forward(self, inputs):
        x = self.conv_in(inputs)
        x1, x2, x3, x4 = torch.split(x, self.cg, dim=1)
        x2 = self.conv_m1(x2)
        x3 = x3 + x2
        x3 = self.conv_m2(x3)
        x4 = x4 + x3
        x4 = self.conv_m3(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_out(x)
        if self.add:
            x = x + inputs
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_block1 = ConvBlock(c1, c2, k=1)
        self.conv_block2 = ConvBlock(c2, c2, k=3)
        self.conv_block3 = ConvBlock(c2, c2, k=3)
        self.add = c1 == c2

    def forward(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        if self.add:
            x = x + inputs
        return x


class FractalBlock(nn.Module):
    def __init__(self, level, c1, c2):
        super().__init__()
        self.level = level
        if level == 1:
            self.block = BottleneckBlock(c1=c1, c2=c2)
        else:
            self.block = nn.Sequential(FractalBlock(level=level - 1, c1=c1, c2=c2),
                                       FractalBlock(level=level - 1, c1=c2, c2=c2))
            self.resblock = BottleneckBlock(c1=c1, c2=c2)

    def forward(self, x):
        if self.level == 1:
            x = self.block(x)
        else:
            x = self.block(x) + self.resblock(x)
        return x


class FractalNet(nn.Module):
    def __init__(self, num_blocks, block_level, input_channel=3, init_filter=32, num_classes=10):
        super().__init__()
        fractal_blocks = []
        in_channel, out_channel = input_channel, init_filter
        for _ in range(num_blocks):
            fractal_blocks.append(FractalBlock(level=block_level, c1=in_channel, c2=out_channel))
            in_channel = out_channel
            out_channel = out_channel * 2
        self.fractal_blocks = nn.ModuleList(fractal_blocks)
        self.fc = nn.Linear(init_filter * 2**(num_blocks - 1), num_classes)

    def forward(self, x):
        for i, fractal_block in enumerate(self.fractal_blocks):
            x = fractal_block(x)
            if i == len(self.fractal_blocks) - 1:
                x = fn.adaptive_avg_pool2d(x, output_size=(1, 1))
            else:
                x = fn.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_estimator(num_blocks, block_level, epochs=200, batch_size=128, save_dir=tempfile.mkdtemp()):
    print("number of blocks: {}, block level: {}".format(num_blocks, block_level))
    # step 1
    train_data, eval_data = cifar10.load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])

    # step 2
    model = fe.build(model_fn=lambda: FractalNet(num_blocks=num_blocks, block_level=block_level, input_channel=3),
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.01, momentum=0.9, weight_decay=0.0001))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    model = FractalNet(num_blocks=3, block_level=3)
    inputs = torch.rand(1, 3, 32, 32)
    pred = model(inputs)
    pdb.set_trace()
