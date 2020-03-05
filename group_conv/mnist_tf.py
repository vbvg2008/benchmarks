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
import tempfile

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.numpy_dataset import NumpyDataset
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class Scale(NumpyOp):
    def forward(self, data, state):
        data = data / 255.0
        return data


def get_estimator(batch_size=32, save_dir=tempfile.mkdtemp()):
    # step 1
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_data = NumpyDataset({"x": train_images, "y": train_labels})
    test_data = NumpyDataset({"x": test_images, "y": test_labels})
    pipeline = fe.Pipeline(train_data=train_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Scale(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=7, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
