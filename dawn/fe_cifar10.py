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
import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.dataset import NumpyDataset
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.metric import Accuracy


def residual(x, num_channel):
    x = layers.Conv2D(num_channel, 3)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def my_model():
    #prep layers
    inp = layers.Input(input_shape=(32, 32, 3))
    x = layers.Conv2D(64, 3, )(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer1
    x = layers.Conv2D(128, 3)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()[x, residual(x, 128)]
    #layer2
    x = layers.Conv2D(256, 3)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer3
    x = layers.Conv2D(512, 3)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()[x, residual(x, 512)]
    #layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


def get_estimator():
    # step 1
    train_data, test_data = load_data()
    pipeline = Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=512,
        ops=Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)))

    # step 2
    model = fe.build(model=my_model(), optimizer="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=24,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
