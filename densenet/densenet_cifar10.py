import pdb
import tempfile

import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import ModelSaver
from fastestimator.trace.metric import Accuracy


def densenet():
    inputs = layers.Input(shape=(32, 32, 3))
    x = tf.keras.applications.DenseNet201(include_top=False, weights=None, input_tensor=inputs, pooling=None)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(epochs=310, batch_size=128, save_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
        ])

    # step 2: prepare network
    model = fe.build(model_fn=densenet, optimizer_fn=lambda: tf.optimizers.SGD(0.05, nesterov=True))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", mode="test"),
        ModelSaver(model=model, save_dir=save_dir, frequency=10),
        LRScheduler(
            model=model,
            lr_fn=lambda epoch: cosine_decay(epoch, cycle_length=10, init_lr=0.05, min_lr=0.001, cycle_multiplier=2))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
