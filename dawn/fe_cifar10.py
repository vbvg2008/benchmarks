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

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import NumpyDataset
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import Normalize
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.trace.metric import Accuracy


def get_estimator():
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    train_data = NumpyDataset({"x": x_train, "y": y_train})
    test_data = NumpyDataset({"x": x_eval, "y": y_eval})
    pipeline = Pipeline(
        train_data=train_data,
        test_data=test_data,
        batch_size=512,
        ops=Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)))

    # step 2
    model = fe.build(model=LeNet(input_shape=(32, 32, 3)), optimizer="adam")
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
