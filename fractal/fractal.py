import pdb

import fastestimator as fe
import tensorflow as tf
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


def conv_block(x, c, k=3):
    x = layers.Conv2D(filters=c, kernel_size=k, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def bottleneck_block(x, c):
    x = layers.Conv2D(filters=c, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=c, kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=c, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)``
    x = layers.ReLU()(x)
    return x


def fractal(x, level, c):
    if level == 1:
        return bottleneck_block(x, c)
    else:
        return fractal(fractal(x, level - 1, c), level - 1, c) + bottleneck_block(x, c)


def mymodel(num_blocks=2, block_level=3, input_shape=(28, 28, 1), init_filter=32, num_classes=10):
    num_filter = init_filter
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = fractal(x, block_level, num_filter)
        x = layers.MaxPool2D()(x)
        num_filter = num_filter * 2
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(epochs=30, batch_size=128):
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
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1)
        ])

    # step 2
    model = fe.build(model_fn=lambda: mymodel(input_shape=(32, 32, 3), num_blocks=4, block_level=4),
                     optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [Accuracy(true_key="y", pred_key="y_pred")]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    mod = mymodel(input_shape=(32, 32, 3), num_blocks=4, block_level=4)
    pdb.set_trace()
    # est = get_estimator(epochs=2)
    # est.fit()
