import pdb
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
from fastestimator.dataset import LabeledDirDataset, mnist
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ExpandDims, Minmax, ReadImage, Resize
from fastestimator.pipeline import Pipeline
from torchvision.models import resnet50


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        data = np.transpose(data, (2, 0, 1))
        return np.float32(data)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    pipeline = Pipeline(
        train_data=LabeledDirDataset("/data/data/ImageNet/train"),
        eval_data=LabeledDirDataset("/data/data/ImageNet/val"),
        batch_size=400,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            Resize(height=224, width=224, image_in="x", image_out="x"),
            Scale(inputs="x", outputs="x")
        ])

    model = resnet50()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = nn.DataParallel(model)
    model.to(device)
    loader = pipeline.get_loader("train", 0)
    criterion = nn.CrossEntropyLoss()
    tic = time.perf_counter()
    i = 0
    for _ in range(5):
        for data in loader:
            # pdb.set_trace()
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
                elapse = time.perf_counter() - tic
                example_sec = 100 / elapse
                print("step: {}, steps/sec: {}".format(i, example_sec))
                tic = time.perf_counter()
