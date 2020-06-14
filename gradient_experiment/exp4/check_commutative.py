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
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

fe.enable_deterministic(1000)


class DebugTrace(fe.trace.Trace):
    def __init__(self, model, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model

    def on_begin(self, data):
        fe.backend.save_model(self.model, save_dir="./", model_name="begin")

    def on_batch_end(self, data):
        gradients = data["gradient"]
        with open("batch32_{}.pkl".format(self.system.batch_idx), 'wb') as f:
            pickle.dump(gradients, f)
        fe.backend.save_model(self.model, save_dir="./", model_name="after_update")
        pdb.set_trace()


class GradientCalculation(fe.op.tensorop.TensorOp):
    def __init__(self, model, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.retain_graph = None

    def forward(self, data, state):
        gradients = fe.backend.get_gradient(data, self.model.trainable_variables, tape=state["tape"])
        return gradients


def get_estimator(epochs=2, max_train_steps_per_epoch=None, max_eval_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train_0, y_train_0 = x_train[np.where(y_train == 0)], y_train[np.where(y_train == 0)]
    x_train_1, y_train_1 = x_train[np.where(y_train == 1)], y_train[np.where(y_train == 1)]
    x_train_2, y_train_2 = x_train[np.where(y_train == 2)], y_train[np.where(y_train == 2)]
    x_train_3, y_train_3 = x_train[np.where(y_train == 3)], y_train[np.where(y_train == 3)]
    ds_train_0 = fe.dataset.NumpyDataset(data={"x": x_train_0, "y": y_train_0})
    ds_train_1 = fe.dataset.NumpyDataset(data={"x": x_train_1, "y": y_train_1})
    ds_train_2 = fe.dataset.NumpyDataset(data={"x": x_train_2, "y": y_train_2})
    ds_train_3 = fe.dataset.NumpyDataset(data={"x": x_train_3, "y": y_train_3})
    train_ds = fe.dataset.BatchDataset(datasets=[ds_train_0, ds_train_1, ds_train_2, ds_train_3],
                                       num_samples=[8, 8, 8, 8])
    x_eval, y_eval = x_eval[np.where(y_eval < 4)], y_eval[np.where(y_eval < 4)]
    eval_ds = fe.dataset.NumpyDataset(data={"x": x_eval, "y": y_eval})
    pipeline = fe.Pipeline(train_data=train_ds,
                           eval_data=eval_ds,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        GradientCalculation(model=model, inputs="ce", outputs="gradient", mode="train"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        DebugTrace(model=model, inputs="gradient", mode="train")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             max_train_steps_per_epoch=max_train_steps_per_epoch,
                             max_eval_steps_per_epoch=max_eval_steps_per_epoch)
    return estimator
