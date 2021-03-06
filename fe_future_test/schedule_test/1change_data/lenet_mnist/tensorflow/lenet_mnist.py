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

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset import NumpyDataset
from fastestimator.op.numpyop import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.pipeline import Pipeline
from fastestimator.schedule import EpochScheduler, RepeatScheduler
from fastestimator.trace.metric import Accuracy


def get_estimator(batch_size=32):
    # step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

    schedule1 = RepeatScheduler(
        repeat_list=[NumpyDataset({
            "x": x_train, "y": y_train
        }), NumpyDataset({
            "x": x_train, "y": y_train
        })])
    schedule2 = EpochScheduler(epoch_dict={
        0: NumpyDataset({
            "x": x_eval, "y": y_eval
        }), 1: NumpyDataset({
            "x": x_eval, "y": y_eval
        })
    })
    schedule3 = RepeatScheduler(repeat_list=[32, 64])
    schedule4 = RepeatScheduler(repeat_list=[Minmax(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])
    pipeline = Pipeline(train_data=schedule1,
                        eval_data=schedule2,
                        batch_size=schedule3,
                        ops=[ExpandDims(inputs="x", outputs="x"), schedule4])
    # pdb.set_trace()
    # step 2
    tf_model = LeNet()
    model1 = fe.build(model=tf_model, optimizer="sgd")
    model2 = fe.build(model=tf_model, optimizer="adam")
    schedule5 = RepeatScheduler(repeat_list=[
        ModelOp(model=model1, inputs="x", outputs="y_pred"), ModelOp(model=model2, inputs="x", outputs="y_pred")
    ])
    schedule6 = RepeatScheduler(repeat_list=[
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"), CrossEntropy(inputs=("y_pred", "y"), outputs="ce")
    ])
    schedule7 = RepeatScheduler(
        repeat_list=[UpdateOp(model=model1, loss_name="ce"), UpdateOp(model=model2, loss_name="ce")])
    network = fe.Network(ops=[schedule5, schedule6, schedule7])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
