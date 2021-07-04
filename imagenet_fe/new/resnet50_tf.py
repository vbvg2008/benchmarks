import os
import pdb
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset import LabeledDirDataset
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, RandomResizedCrop, Resize
from fastestimator.op.numpyop.numpyop import LambdaOp
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule import EpochScheduler
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, RestoreWizard
from fastestimator.trace.metric import Accuracy
from fastestimator.util import get_num_devices
from tensorflow.keras import layers, mixed_precision

mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))


def center_crop(data):
    image_height, image_width, _ = data.shape
    padded_center_crop_size = int(224 / (224 + 32) * min(image_height, image_height))
    y1 = ((image_height - padded_center_crop_size) + 1) // 2
    x1 = ((image_width - padded_center_crop_size) + 1) // 2
    return data[y1:y1 + padded_center_crop_size, x1:x1 + padded_center_crop_size, :]


class RGBScale(fe.op.tensorop.TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.mean_rgb = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], shape=[1, 1, 3], dtype=tf.float32)
        self.std_rgb = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], shape=[1, 1, 3], dtype=tf.float32)

    def forward(self, data, state):
        image = tf.cast(data, tf.float32)
        return (image - self.mean_rgb) / self.std_rgb


class L2Loss(fe.op.tensorop.TensorOp):
    def __init__(self, model, weight_decay, inputs, outputs, mode="train"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.weight_decay = weight_decay

    def forward(self, data, state):
        l2_loss = tf.reduce_sum(
            [tf.nn.l2_loss(tf.cast(v, tf.float16)) for v in self.model.trainable_variables if 'bn' not in v.name])
        return data + self.weight_decay * l2_loss


def lr_warmup_fn(step, init_lr, steps_per_epoch, warmup_epochs=10):
    warmup_steps = steps_per_epoch * warmup_epochs
    return init_lr * step / warmup_steps


def lr_decay_fn(epoch, init_lr):
    if epoch <= 60:
        lr = init_lr
    elif epoch <= 120:
        lr = init_lr * 0.1
    elif epoch <= 160:
        lr = init_lr * 0.01
    else:
        lr = init_lr * 0.001
    return lr


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    if strides > 1:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return layers.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same' if strides == 1 else 'valid',
                         use_bias=False,
                         kernel_initializer=tf.keras.initializers.VarianceScaling())(inputs)


def batchnorm_relu(x, nonlinearity=True, init_zero=False, bn_momentum=0.9, epsilon=1e-5):
    x = layers.BatchNormalization(momentum=bn_momentum,
                                  epsilon=epsilon,
                                  gamma_initializer='zeros' if init_zero else 'ones')(x)
    if nonlinearity:
        x = tf.nn.relu(x)
    return x


def block_fn(inputs, filters, strides, use_projection, bn_momentum):
    if use_projection:
        filters_out = 4 * filters
        shortcut = conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)
        shortcut = batchnorm_relu(shortcut, nonlinearity=False, bn_momentum=bn_momentum)
    else:
        shortcut = inputs
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1)
    inputs = batchnorm_relu(inputs, bn_momentum=bn_momentum)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)
    inputs = batchnorm_relu(inputs, bn_momentum=bn_momentum)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
    inputs = batchnorm_relu(inputs, nonlinearity=False, init_zero=True, bn_momentum=bn_momentum)
    return tf.nn.relu(inputs + shortcut)


def block_group_fn(x, filters, blocks, strides, bn_momentum):
    x = block_fn(x, filters=filters, strides=strides, use_projection=True, bn_momentum=bn_momentum)
    for _ in range(1, blocks):
        x = block_fn(x, filters=filters, strides=1, use_projection=False, bn_momentum=bn_momentum)
    return x


def official_resnet50(bn_momentum=0.9, num_layers=(3, 4, 6, 3)):
    inputs = layers.Input(shape=(224, 224, 3))
    x = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2)
    x = batchnorm_relu(x, bn_momentum=bn_momentum)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = block_group_fn(x, filters=64, blocks=num_layers[0], strides=1, bn_momentum=bn_momentum)
    x = block_group_fn(x, filters=128, blocks=num_layers[1], strides=2, bn_momentum=bn_momentum)
    x = block_group_fn(x, filters=256, blocks=num_layers[2], strides=2, bn_momentum=bn_momentum)
    x = block_group_fn(x, filters=512, blocks=num_layers[3], strides=2, bn_momentum=bn_momentum)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.01))(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(data_dir, batch_per_gpu=256, epochs=180, save_dir=tempfile.mkdtemp(), restore_dir=tempfile.mkdtemp()):

    train_ds = LabeledDirDataset(os.path.join(data_dir, "train"))
    eval_ds = LabeledDirDataset(os.path.join(data_dir, "val"))
    batch_size = batch_per_gpu * get_num_devices()
    pipeline = Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="x", outputs="x"),
            RandomResizedCrop(height=224,
                              width=224,
                              image_in="x",
                              image_out="x",
                              interpolation=cv2.INTER_CUBIC,
                              scale=(0.1, 1.0),
                              mode="train"),
            LambdaOp(fn=center_crop, inputs="x", outputs="x", mode="eval"),
            Resize(height=224, width=224, image_in="x", image_out="x", interpolation=cv2.INTER_CUBIC, mode="eval"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
        ])
    # step 2
    init_lr = 0.1 * batch_size / 256
    model = fe.build(model_fn=official_resnet50,
                     optimizer_fn=lambda: tf.optimizers.SGD(learning_rate=init_lr, momentum=0.9, nesterov=True),
                     mixed_precision=True)
    network = fe.Network(ops=[
        RGBScale(inputs="x", outputs="x"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", from_logits=True),
        L2Loss(model=model, weight_decay=1e-4, inputs="ce", outputs="ce", mode="train"),
        UpdateOp(model=model, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        RestoreWizard(directory=restore_dir)
    ]
    lr_schedule = {
        1:
        LRScheduler(
            model=model,
            lr_fn=lambda step: lr_warmup_fn(step, init_lr=init_lr, steps_per_epoch=np.ceil(len(train_ds) / batch_size))
        ),
        11:
        LRScheduler(model=model, lr_fn=lambda epoch: lr_decay_fn(epoch, init_lr=init_lr))
    }
    traces.append(EpochScheduler(lr_schedule))
    # step 3
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
