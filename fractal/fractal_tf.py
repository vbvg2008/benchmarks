import pdb
import tempfile

import fastestimator as fe
import tensorflow as tf
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


def conv_block(x, c, k=3):
    x = layers.Conv2D(filters=c, kernel_size=k, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)
    return x


def bottleneck_block(inputs, c):
    x = conv_block(inputs, c, k=1)
    x = conv_block(x, c, k=3)
    x = conv_block(x, c, k=1)
    if x.shape[-1] == inputs.shape[-1]:
        outputs = x + inputs
    else:
        outputs = x
    return outputs


def res2net_block(inputs, c, s=4):
    cg = c // s
    x = conv_block(inputs, c, k=1)
    x1, x2, x3, x4 = x[..., :cg], x[..., cg:2 * cg], x[..., 2 * cg:3 * cg], x[..., 3 * cg:]
    x2 = conv_block(x2, cg, k=3)
    x3 = x3 + x2
    x3 = conv_block(x3, cg, k=3)
    x4 = x4 + x3
    x4 = conv_block(x4, cg, k=3)
    x = tf.concat([x1, x2, x3, x4], axis=-1)
    x = conv_block(x, c, k=1)
    if x.shape[-1] == inputs.shape[-1]:
        outputs = x + inputs
    else:
        outputs = x
    return outputs


def fractal(x, level, c):
    if level == 1:
        return bottleneck_block(x, c)
    else:
        return fractal(fractal(x, level - 1, c), level - 1, c) + bottleneck_block(x, c)


def mymodel(num_blocks=2, block_level=3, input_shape=(28, 28, 1), init_filter=32, num_classes=10):
    num_filter = init_filter
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = fractal(x, block_level, num_filter)
        if i == num_blocks - 1:
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.MaxPool2D()(x)
        num_filter = num_filter * 2
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(num_blocks, block_level, epochs=200, batch_size=128, save_dir=tempfile.mkdtemp()):
    print("number of blocks: {}, block level: {}".format(num_blocks, block_level))
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
    model = fe.build(model_fn=lambda: mymodel(input_shape=(32, 32, 3), num_blocks=num_blocks, block_level=block_level),
                     optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
