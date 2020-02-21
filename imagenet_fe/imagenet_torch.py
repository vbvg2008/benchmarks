import pdb

import cv2
import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import NumpyDataset
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ReadImage, Resize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.metric import Accuracy
from label_dir_dataset import LabeledDirDataset
from torchvision.models import inception_v3


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        data = np.transpose(data, (2, 0, 1))
        return np.float32(data)


def get_estimator():
    pipeline = Pipeline(
        train_data=LabeledDirDataset("/data/data/ImageNet/train"),
        eval_data=LabeledDirDataset("/data/data/ImageNet/val"),
        batch_size=128,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            Resize(height=299, width=299, image_in="x", image_out="x"),
            Scale(inputs="x", outputs="x")
        ])

    model = fe.build(model=inception_v3(aux_logits=False), optimizer="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", apply_softmax=True),
        UpdateOp(model=model, loss_name="ce")
    ])

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
