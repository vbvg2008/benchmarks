import pdb

import fastestimator as fe
import tensorflow as tf
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import TensorBoard
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


def conv_block(x, c, k=1, s=1):
    x = layers.Conv2D(filters=c, kernel_size=k, strides=s, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.97)(x)
    x = tf.nn.relu(x)
    return x


def fractal(level, x):
    if level == 1:
        return conv_block(x, 32, k=3)
    else:
        return fractal(level - 1, fractal(level - 1, x)) + conv_block(x, 32, k=3)


def mymodel(levle=3, input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = fractal(levle, inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(epochs=2, batch_size=32):
    # step 1
    train_data, eval_data = mnist.load_data()
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=mymodel, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        TensorBoard(log_dir="/data/Xiaomeng/fractalreport", weight_histogram_freq="epoch")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator(epochs=2)
    est.fit()
