# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This example showcase FastEstimator usage for tensorflow users. In this file, we use tf.dataset as data input.
"""
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras import backend, layers
from tensorflow.python.keras.regularizers import l2

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import CoarseDropout, HorizontalFlip, Normalize, PadIfNeeded, RandomCrop, Sometimes
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace import Trace
from fastestimator.trace.metric import Accuracy

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

Batch = 512
Epoch = 24
Warmup = 5
STEPS_PER_EPOCH = 50000 // Batch + 1


class SmootOneHot(NumpyOp):
    def forward(self, data, state):
        output = np.full((10), fill_value=0.2 / 9)
        output[data[0]] = 0.8
        return output


class LRChange(Trace):
    def __init__(self, model):
        super().__init__(mode="train")
        self.model = model

    def on_batch_begin(self, data):
        if self.system.global_step > Warmup * STEPS_PER_EPOCH:
            lr = (Epoch * STEPS_PER_EPOCH - self.system.global_step) / ((Epoch - Warmup) * STEPS_PER_EPOCH) * 0.4
        else:
            lr = self.system.global_step / (Warmup * STEPS_PER_EPOCH) * 0.4
        backend.set_value(self.model.optimizer.lr, lr)


def residual(x, num_channel, c=5e-4 * Batch):
    x = layers.Conv2D(num_channel, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def my_model(c=5e-4 * Batch):
    #prep layers
    inp = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer1
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    #layer2
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer3
    x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8, beta_regularizer=l2(c), gamma_regularizer=l2(c))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    #layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, kernel_regularizer=l2(c), bias_regularizer=l2(c))(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


def get_estimator():
    # step 1
    train_data, test_data = load_data()
    pipeline = Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=Batch,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            SmootOneHot(inputs="y", outputs="y", mode="train")
        ])
    # step 2
    model = fe.build(model=my_model(), optimizer="adam")
    # pdb.set_trace()
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=Epoch,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
