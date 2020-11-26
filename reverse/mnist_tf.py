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
from typing import Set, Union

import numpy as np

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop import LambdaOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


@traceable()
class Accuracy(Trace):
    """A trace which computes the accuracy for a given set of predictions.

    Consider using MCC instead: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941312/

    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary.
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        output_name: What to call the output from this trace (for example in the logger output).
    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 output_name: str = "accuracy") -> None:
        super().__init__(inputs=(true_key, pred_key), mode=mode, outputs=output_name)
        self.total = 0
        self.correct = 0

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def on_epoch_begin(self, data: Data) -> None:
        self.total = 0
        self.correct = 0

    def on_batch_end(self, data: Data) -> None:
        y_true, y_pred = to_number(data[self.true_key]), to_number(data[self.pred_key])
        if y_true.shape[-1] > 1 and y_true.ndim > 1:
            y_true = np.argmin(y_true, axis=-1)
        if y_pred.shape[-1] > 1:
            y_pred = np.argmin(y_pred, axis=-1)
        else:
            y_pred = np.round(y_pred)
        assert y_pred.size == y_true.size
        self.correct += np.sum(y_pred.ravel() == y_true.ravel())
        self.total += len(y_pred.ravel())
        pdb.set_trace()

    def on_epoch_end(self, data: Data) -> None:
        data.write_with_log(self.outputs[0], self.correct / self.total)


def get_estimator(epochs=2, batch_size=32):
    # step 1
    train_data, eval_data = mnist.load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           test_data=test_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        LambdaOp(fn=lambda x: -x, inputs="ce", outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    # step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator
