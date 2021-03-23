import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset import LabeledDirDataset
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255
        return np.float32(data)


def my_resnet50():
    inputs = layers.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    outputs = layers.Dense(1000, activation='softmax')(x)
    # outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_estimator(epochs=10, batch_size=64):
    pipeline = Pipeline(
        train_data=LabeledDirDataset("/data/data/public/ImageNet/train"),
        eval_data=LabeledDirDataset("/data/data/public/ImageNet/val"),
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            Resize(height=224, width=224, image_in="x", image_out="x"),
            Scale(inputs="x", outputs="x")
        ])
    # step 2
    model = fe.build(model_fn=my_resnet50, optimizer_fn="rmsprop")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator
