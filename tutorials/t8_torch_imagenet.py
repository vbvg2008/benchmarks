import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torchvision.models import resnet50

from fastestimator.dataset import LabeledDirDataset, mnist
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ExpandDims, Minmax, ReadImage, Resize
from fastestimator.pipeline import Pipeline


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        data = np.transpose(data, (2, 0, 1))
        return np.float32(data)


def new_forward(data, model):
    pdb.set_trace()
    replicas = nn.parallel.replicate(model, [0, 1, 2, 3])
    x = nn.parallel.scatter(data["x"], [0, 1, 2, 3])
    y = nn.parallel.scatter(data["y"], [0, 1, 2, 3])
    y_pred = nn.parallel.parallel_apply(replicas, x)
    loss = nn.parallel.parallel_apply(criterion(y_pred, y.long()))
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")
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

    # model = nn.DataParallel(model)
    model.to(device0)
    criterion = nn.CrossEntropyLoss()

    loader = pipeline.get_loader("train", 0)
    tic = time.perf_counter()
    i = 0
    for _ in range(5):
        for data in loader:
            new_forward(data, model)
            # #zero the parameter gradients
            optimizer.zero_grad()
            # y_pred = model(x)
            # loss = criterion(y_pred, y.long())
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
