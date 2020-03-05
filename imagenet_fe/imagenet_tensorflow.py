import pdb

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# from labeled_dir_dataset import LabeledDirDataset
from tensorflow.keras.mixed_precision import experimental as mixed_precision

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

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def my_inceptionv3():
    inputs = layers.Input(shape=(299, 299, 3))
    backbone = tf.keras.applications.InceptionV3(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    x = layers.Dense(1000)(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def my_resnet50():
    inputs = layers.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    x = layers.Dense(1000)(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        return np.float32(data)


def get_estimator():
    pipeline = Pipeline(
        train_data=LabeledDirDataset("/data/data/ImageNet/train"),
        eval_data=LabeledDirDataset("/data/data/ImageNet/val"),
        batch_size=256,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            Resize(height=224, width=224, image_in="x", image_out="x"),
            Scale(inputs="x", outputs="x")
        ])

    # step 2
    model = fe.build(model_fn=my_resnet50, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
