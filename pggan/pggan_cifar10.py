import os
import tempfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Loss, ModelOp, Resize
from fastestimator.schedule import Scheduler
from fastestimator.trace import ModelSaver, Trace
from pggan_architecture import build_D, build_G


class Rescale(TensorOp):
    """Scale image values from uint8 to float32 between -1 and 1."""
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


class RandomInput(TensorOp):
    def __init__(self, latent_dim, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.latent_dim = latent_dim

    def forward(self, data, state):
        batch_size = state["local_batch_size"]
        rv = tf.random.normal([batch_size, self.latent_dim])
        return tf.math.l2_normalize(rv, axis=-1)


class Interpolate(TensorOp):
    def forward(self, data, state):
        fake, real = data
        fake = tf.cast(fake, tf.float32)
        real = tf.cast(real, tf.float32)
        batch_size = state["local_batch_size"]
        coeff = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
        return real + (fake - real) * coeff


class GradientPenalty(TensorOp):
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        interp, d_interp = data
        d_interp = tf.reshape(d_interp, [-1])
        tape = state['tape']

        d_interp_loss = tf.reduce_sum(d_interp)
        grad_interp = tape.gradient(d_interp_loss, interp)
        grad_l2 = tf.math.sqrt(tf.reduce_sum(tf.math.square(grad_interp), axis=[1, 2, 3]))
        gp = tf.math.square(grad_l2 - 1)
        return gp


class GLoss(Loss):
    """Compute generator loss."""
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        fake = data
        loss = tf.reshape(fake, [-1])
        return loss


class DLoss(Loss):
    """Compute discriminator loss."""
    def __init__(self, inputs, outputs=None, mode=None, LAMBDA=10):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.LAMBDA = LAMBDA

    def forward(self, data, state, eps=0.001):
        d_real, d_fake, gp = data
        d_real = tf.reshape(d_real, [-1])
        d_fake = tf.reshape(d_fake, [-1])
        loss = d_real - d_fake + self.LAMBDA * gp + d_real * eps
        return loss


class AlphaController(Trace):
    def __init__(self, epoch_schedule, interval, nimg=600, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.epoch_schedule = epoch_schedule
        self.interval = interval
        self.change_alpha = False
        self._idx = 0
        self.model_name = None

        self.nimg_so_far = 0.0
        self.nimg_total = tf.convert_to_tensor(nimg, dtype=tf.float32)

    def on_epoch_begin(self, state):
        # check whetehr the current epoch is in smooth transition of resolutions
        current_key = list(self.epoch_schedule.keys())[self._idx]
        if state["epoch"] == current_key:
            self.change_alpha = True
            self.model_name = self.epoch_schedule[current_key]
            self.nimg_so_far = 0.0
            print("FastEstimator-Alpha: Started fading in %s\n" % (self.network.model[self.model_name].model_name))
        else:
            if state["epoch"] == current_key + self.interval:
                print("FastEstimator-Alpha: Finished fading in %s\n" % (self.network.model[self.model_name].model_name))
                self.change_alpha = False
                self._idx += 1

    def on_batch_begin(self, state):
        # if in resolution transition, smoothly change the alpha from 0 to 1
        if self.change_alpha:
            self.nimg_so_far += state["batch_size"] / 1000
            curr_alpha = tf.clip_by_value(self.nimg_so_far / self.nimg_total, 0, 1)
            self.network.model[self.model_name].alpha.assign(curr_alpha)


def get_estimator():
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    data = {"train": {"x": x}}
    # We create a scheduler for batch_size with the epochs at which it will change and corresponding values.
    batchsize_scheduler = Scheduler({0: 64, 50: 32})

    # We create a scheduler for the Resize ops.
    resize_scheduler = Scheduler({
        0: Resize(inputs="x", size=(4, 4), outputs="x"),
        10: Resize(inputs="x", size=(8, 8), outputs="x"),
        30: Resize(inputs="x", size=(16, 16), outputs="x"),
        50: Resize(inputs="x", size=(32, 32), outputs="x")
    })

    # In Pipeline, we use the schedulers for batch_size and ops.
    pipeline = fe.Pipeline(batch_size=batchsize_scheduler,
                           data=data,
                           ops=[resize_scheduler, Rescale(inputs="x", outputs="x")])

    d2, d3, d4, d5 = fe.build(model_def=lambda:build_D(target_resolution=5),
                            model_name=["d2", "d3", "d4", "d5"],
                            optimizer=[
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)],
                            loss_name=["d_loss", "d_loss", "d_loss", "d_loss"])

    g2, g3, g4, g5, G = fe.build(model_def=lambda:build_G(target_resolution=5),
                            model_name=["g2", "g3", "g4", "g5", "G"],
                            optimizer=[
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
                                tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)],
                            loss_name=["g_loss", "g_loss", "g_loss", "g_loss", "g_loss"])

    noise_scheduler = Scheduler({
        0: RandomInput(inputs="x", latent_dim=512, outputs="z"),
        50: RandomInput(inputs="x", latent_dim=512, outputs="z"),
    })
    g_scheduler = Scheduler({
        0: ModelOp(inputs="z", model=g2, outputs="x_fake"),
        10: ModelOp(inputs="z", model=g3, outputs="x_fake"),
        30: ModelOp(inputs="z", model=g4, outputs="x_fake"),
        50: ModelOp(inputs="z", model=g5, outputs="x_fake")
    })

    d_fake_scheduler = Scheduler({
        0: ModelOp(inputs="x_fake", model=d2, outputs="d_fake"),
        10: ModelOp(inputs="x_fake", model=d3, outputs="d_fake"),
        30: ModelOp(inputs="x_fake", model=d4, outputs="d_fake"),
        50: ModelOp(inputs="x_fake", model=d5, outputs="d_fake"),
    })

    d_real_scheduler = Scheduler({
        0: ModelOp(inputs="x", model=d2, outputs="d_real"),
        10: ModelOp(inputs="x", model=d3, outputs="d_real"),
        30: ModelOp(inputs="x", model=d4, outputs="d_real"),
        50: ModelOp(inputs="x", model=d5, outputs="d_real"),
    })

    d_interp_scheduler = Scheduler({
        0: ModelOp(inputs="x_interp", model=d2, outputs="d_interp", track_input=True),
        10: ModelOp(inputs="x_interp", model=d3, outputs="d_interp", track_input=True),
        30: ModelOp(inputs="x_interp", model=d4, outputs="d_interp", track_input=True),
        50: ModelOp(inputs="x_interp", model=d5, outputs="d_interp", track_input=True),
    })

    network = fe.Network(ops=[
        noise_scheduler,
        g_scheduler,
        d_fake_scheduler,
        d_real_scheduler,
        Interpolate(inputs=("x_fake", "x"), outputs="x_interp"),
        d_interp_scheduler,
        GradientPenalty(inputs=("x_interp", "d_interp"), outputs="gp"),
        GLoss(inputs="d_fake", outputs="gloss"),
        DLoss(inputs=("d_real", "d_fake", "gp"), outputs="dloss")
    ])

    alpha_model = {10: ["g3", "d3"], 30: ["g4", "d4"], 50: ["g5", "d5"]}

    traces = [
        AlphaController(epoch_schedule=alpha_model, interval=10), ]

    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=60, traces=traces)
    return estimator
