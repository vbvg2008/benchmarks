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


def get_estimator(epochs=2,
                  batch_size=32,
                  max_train_steps_per_epoch=None,
                  max_eval_steps_per_epoch=None,
                  save_dir=tempfile.mkdtemp()):
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train_0, y_train_0 = x_train[np.where(y_train == 0)], y_train[np.where(y_train == 0)]
    x_train_1, y_train_1 = x_train[np.where(y_train == 1)], y_train[np.where(y_train == 1)]
    ds_train_0 = fe.dataset.NumpyDataset(data={"x": x_train_0, "y": y_train_0})
    ds_train_1 = fe.dataset.NumpyDataset(data={"x": x_train_1, "y": y_train_1})
    train_ds = fe.dataset.BatchDataset(datasets=[ds_train_0, ds_train_1], num_samples=[16, 16])
    x_eval, y_eval = x_eval[np.where(y_eval < 2)], y_eval[np.where(y_eval < 2)]
    eval_ds = fe.dataset.NumpyDataset(data={"x": x_eval, "y": y_eval})
    pipeline = fe.Pipeline(train_data=train_ds,
                           eval_data=eval_ds,
                           batch_size=32,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

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
        # DebugTrace(inputs=["gradient", "gradient1", "gradient2"], mode="train")
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
