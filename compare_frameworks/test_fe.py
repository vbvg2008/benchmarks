import datetime
import tempfile

import fastestimator as fe
import tensorflow as tf
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from test_base import batch_size, epochs, my_model_tf, x_eval, x_train, y_eval, y_train

train_data = fe.dataset.NumpyDataset({"x": x_train, "y": y_train})
eval_data = fe.dataset.NumpyDataset({"x": x_eval, "y": y_eval})

pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, batch_size=batch_size,
                       ops=[])  #ExpandDims(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

model = fe.build(model_fn=my_model_tf, optimizer_fn=lambda: tf.optimizers.Adam(1e-4))

network = fe.Network(ops=[
    ModelOp(model=model, inputs="x", outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce")
])

traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    BestModelSaver(model=model, save_dir=tempfile.mkdtemp(), metric="accuracy", save_best_mode="max")
]

estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=[])  #traces)

startTime = datetime.datetime.now()

estimator.fit()

print("Elapsed seconds {}".format(datetime.datetime.now() - startTime))
