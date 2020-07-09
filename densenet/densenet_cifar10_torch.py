import tempfile

import torch
import torchvision

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ChannelTranspose, CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import ModelSaver
from fastestimator.trace.metric import Accuracy


class DenseNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        densenet_layers = list(torchvision.models.densenet201(pretrained=False).children())
        self.densenet_feature = torch.nn.Sequential(densenet_layers[0])
        self.classifier = torch.nn.Linear(1920, num_classes)

    def forward(self, x):
        x = self.densenet_feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x


def get_estimator(epochs=310, batch_size=128, save_dir=tempfile.mkdtemp(), max_train_steps_per_epoch=None):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            ChannelTranspose(inputs="x", outputs="x")
        ])

    # step 2: prepare network
    model = fe.build(model_fn=DenseNet, optimizer_fn=lambda x: torch.optim.SGD(x, lr=0.05))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", mode="eval"),
        ModelSaver(model=model, save_dir=save_dir, frequency=10),
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=10, init_lr=0.05, min_lr=0.001, cycle_multiplier=2))
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch)
    return estimator
