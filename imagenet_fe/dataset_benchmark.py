import pdb

import cv2
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import LabeledDirDataset, NumpyDataset
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ExpandDims, Minmax, ReadImage, Resize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.metric import Accuracy

# from label_dir_dataset import LabeledDirDataset

# from labeled_dir_dataset import LabeledDirDataset


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        return np.float32(data)


if __name__ == "__main__":
    pipeline = Pipeline(
        train_data=LabeledDirDataset("/data/data/ImageNet/train"),
        eval_data=LabeledDirDataset("/data/data/ImageNet/val"),
        batch_size=64,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            Resize(height=299, width=299, image_in="x", image_out="x"),
            Scale(inputs="x", outputs="x")
        ])

    loader = pipeline.get_loader("train", 0)

    # data_1 = next(iter(loader))
    tf_data = tf.data.Dataset.from_generator(lambda: loader, {"x": tf.float32, "y": tf.int64})
    batch = next(tf_data.__iter__)
    # data_2 = next(iter(tf_data))
    pdb.set_trace()
    # pipeline2 = Pipeline(train_data=tf_data)
    # pipeline2.benchmark()
