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
import pdb
import pickle
import tempfile

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

fe.enable_deterministic(1000)


class DebugTrace(fe.trace.Trace):
    def on_batch_end(self, data):
        gradients = data["gradient"]
        gradients1 = data["gradient1"]
        gradients2 = data["gradient2"]
        with open("batch32_1.pkl", 'wb') as f:
            pickle.dump(gradients, f)
        gradients = data["gradient"]
        with open("batch16_1.pkl", 'wb') as f:
            pickle.dump(gradients1, f)
        gradients = data["gradient"]
        with open("batch16_2.pkl", 'wb') as f:
            pickle.dump(gradients2, f)
        pdb.set_trace()


class GradientCalculation(fe.op.tensorop.TensorOp):
    def __init__(self, model, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.retain_graph = None

    def forward(self, data, state):
        gradients = fe.backend.get_gradient(data, self.model.trainable_variables, tape=state["tape"])
        return gradients


class SplitData(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        return data[:16], data[16:]


def get_estimator(epochs=2,
                  batch_size=32,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train_1, y_train_1 = x_train[np.where(y_train == 1)], y_train[np.where(y_train == 1)]
    x_train_2, y_train_2 = x_train[np.where(y_train == 2)], y_train[np.where(y_train == 2)]
    ds_train_1 = fe.dataset.NumpyDataset(data={"x": x_train_1, "y": y_train_1})
    ds_train_2 = fe.dataset.NumpyDataset(data={"x": x_train_2, "y": y_train_2})
    train_ds = fe.dataset.BatchDataset(datasets=[ds_train_1, ds_train_2], num_samples=[16, 16])

    pipeline = fe.Pipeline(train_data=train_ds,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        SplitData(inputs="x", outputs=["x1", "x2"]),
        SplitData(inputs="y", outputs=["y1", "y2"]),
        ModelOp(model=model, inputs="x1", outputs="y_pred1"),
        ModelOp(model=model, inputs="x2", outputs="y_pred2"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred1", "y1"), outputs="ce1"),
        CrossEntropy(inputs=("y_pred2", "y2"), outputs="ce2"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        GradientCalculation(model=model, inputs="ce", outputs="gradient", mode="train"),
        GradientCalculation(model=model, inputs="ce1", outputs="gradient1", mode="train"),
        GradientCalculation(model=model, inputs="ce2", outputs="gradient2", mode="train")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=3750, init_lr=1e-3)),
        DebugTrace(inputs=["gradient", "gradient1", "gradient2"], mode="train")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
