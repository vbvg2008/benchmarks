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
import os
import pdb
import tempfile

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import LeNet
from fastestimator.dataset.mnist import load_data
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ImageReader, Resize
from fastestimator.op.tensorop import Minmax, ModelOp, Reshape, SparseCategoricalCrossentropy
from fastestimator.schedule.epoch_scheduler import Scheduler
from fastestimator.trace import Accuracy, ModelSaver


class ExtractLabel(TensorOp):
    def forward(self, data, state):
        # data is batch x padded
        data = data[:, 0]
        return data


class VaryLabel(NumpyOp):
    def forward(self, data, state):
        varied_list = [data]
        length = np.random.randint(1, 11)
        for _ in range(length):
            varied_list.append(0)
        return varied_list, length + 1


def get_estimator(epochs=2, batch_size=2):
    # step 1. prepare data
    train_csv, eval_csv, path = load_data()
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "tfrecord_var_label"),
        train_data=train_csv,
        validation_data=eval_csv,
        ops=[
            ImageReader(inputs="x", grey_scale=True, parent_path=path, outputs="x"),
            VaryLabel(inputs="y", outputs=["y", "y_length"])
        ])
    pipeline = fe.Pipeline(batch_size=batch_size, data=writer, ops=Minmax(inputs="x", outputs="x"), padded_batch=True)

    # step 2. prepare model
    model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="loss")
    network = fe.Network(ops=[
        ExtractLabel(inputs="y", outputs="y"),
        # Reshape(shape=(2, 28, 28, 1), inputs="x"),
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss")
    ])

    # step 3.prepare estimator
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=Accuracy(true_key="y", pred_key="y_pred", output_name='acc'))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
