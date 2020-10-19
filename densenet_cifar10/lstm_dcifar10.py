import pdb
import tempfile

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.backend import get_gradient, get_lr
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize, Onehot
from fastestimator.op.tensorop.gradient import GradientOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from tensorflow.python.keras import layers

from densenet import createDenseNet


def my_lstm():
    # prep layers
    inp = layers.Input(shape=(1, 1), batch_size=1)
    x = layers.LSTM(20, return_sequences=True, stateful=True)(inp)
    x = layers.LSTM(1, stateful=True)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


class CreateModelParam(fe.op.tensorop.TensorOp):
    def __init__(self, model, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model

    def forward(self, data, state):
        return self.model.trainable_variables


class DebugOp(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        pdb.set_trace()
        return data


class CosineSimilarity(fe.op.tensorop.TensorOp):
    def __init__(self, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction='none')
        self.num_params = 1002586

    def forward(self, data, state):
        grad_previous, grad_now = data
        length = len(grad_now)
        similarity = tf.convert_to_tensor(0.0, dtype=tf.float32)
        if grad_previous is not None and isinstance(grad_previous, list):
            for idx in range(length):
                grad1, grad2 = grad_previous[idx], grad_now[idx]
                grad1 = tf.reshape(grad1, (-1, 1))
                grad2 = tf.reshape(grad2, (-1, 1))
                similarity += -tf.reduce_sum(self.loss(grad1, grad2))
            similarity = similarity / self.num_params
        return tf.reshape(similarity, (1, 1, 1))


class DataProvider(fe.trace.Trace):
    def __init__(self, inputs, outputs, mode):
        super().__init__(inputs, outputs, mode=mode)
        self.previous_grad = None

    def on_batch_begin(self, data):
        data.write_without_log("previous_grad", self.previous_grad)

    def on_batch_end(self, data):
        self.previous_grad = data["gradient"]
        cosine_similarity = data["cosine_similarity"]


class ShowLR(fe.trace.Trace):
    def __init__(self, model):
        super().__init__(inputs=None, outputs=None, mode="train")
        self.model = model

    def on_batch_end(self, data):
        if self.system.global_step % self.system.log_steps == 0 or self.system.global_step == 1:
            data.write_with_log("model_lr", np.float32(get_lr(self.model)))


class GradientUpdate(fe.op.tensorop.TensorOp):
    def __init__(self, model, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model

    def forward(self, data, state):
        lr, gradients = data
        lr = tf.reshape(lr, ())
        # lstm may predict negative lrs, cliping the lr
        lr = tf.maximum(lr, 0.0)
        self.model.current_optimizer.lr.assign(lr)
        self.model.current_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class GradientUpdate2(fe.op.tensorop.TensorOp):
    def __init__(self, model, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model

    def forward(self, data, state):
        gradients1, gradients2, lr = data
        gradients3 = get_gradient(lr, self.model.trainable_variables, tape=state["tape"])
        grad1dotgrad2 = 0.0
        for idx in range(len(gradients1)):
            grad1 = tf.reshape(gradients1[idx], (-1))
            grad2 = tf.reshape(gradients2[idx], (-1))
            grad1dotgrad2 += tf.tensordot(-grad1, grad2, axes=1)
        for idx, grad in enumerate(gradients3):
            gradients3[idx] = grad * grad1dotgrad2
        # update lstm
        self.model.current_optimizer.apply_gradients(zip(gradients3, self.model.trainable_variables))
        # pdb.set_trace()


class FillData(fe.op.numpyop.NumpyOp):
    def forward(self, data, state):
        return 0.0


def get_estimator(epochs=13, batch_size=128):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        # test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
            # FillData(inputs=None, outputs="previous_grad")
            # PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            # RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            # Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            # CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
            # Onehot(inputs="y", outputs="y", mode="train", num_classes=10, label_smoothing=0.2)
        ])

    # step 2: prepare network
    model = fe.build(model_fn=lambda: createDenseNet(10, (32, 32, 3)), optimizer_fn="sgd")
    lstm_model = fe.build(model_fn=my_lstm, optimizer_fn=lambda: tf.optimizers.Adam(0.01))
    network = fe.Network(ops=[
        CreateModelParam(model=model, inputs=None, outputs="model_params", mode="train"),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss_before"),
        GradientOp(inputs="model_params", finals="loss_before", outputs="gradient"),
        CosineSimilarity(inputs=("previous_grad", "gradient"), outputs="cosine_similarity", mode="train"),
        ModelOp(model=lstm_model, inputs="cosine_similarity", outputs="lr"),
        GradientUpdate(model=model, inputs=("lr", "gradient"), outputs=None, mode="train"),
        ModelOp(model=model, inputs="x", outputs="y_pred2"),
        CrossEntropy(inputs=("y_pred2", "y"), outputs="loss_after"),
        GradientOp(inputs="model_params", finals="loss_after", outputs="gradient2"),
        GradientUpdate2(model=lstm_model, inputs=("gradient", "gradient2", "lr"), outputs=None, mode="train")
        # DebugOp(inputs="gradient", outputs="gradient")
    ])

    # step 3: prepare estimator
    traces = [
        DataProvider(inputs=("gradient", "cosine_similarity"), outputs="previous_grad", mode="train"),
        ShowLR(model=model),
        Accuracy(true_key="y", pred_key="y_pred")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names="loss_before",
                             log_steps=100)
    return estimator


if __name__ == "__main__":
    model = my_lstm()
