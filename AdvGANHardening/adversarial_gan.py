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
from typing import Iterable, Set, Union

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op import LambdaOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import CoarseDropout, Normalize
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, GradientOp, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler
from fastestimator.trace.adapt import EarlyStopping, ReduceLROnPlateau
from fastestimator.trace.io import BestModelSaver, Traceability
from fastestimator.trace.metric import Accuracy, MCC
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number
from tensorflow.python.keras import Model, layers
from tensorflow.python.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model


@traceable()
class Confidence(Trace):
    def __init__(self,
                 key: str,
                 mode: Union[str, Set[str]] = ("eval", "test"),
                 output_name: Iterable[str] = ("conf_mean", "conf_std", "conf_median", "conf_thres")) -> None:
        super().__init__(inputs=key, mode=mode, outputs=output_name)
        self.max_probs = []

    def on_epoch_begin(self, data: Data) -> None:
        self.max_probs = []

    def on_batch_end(self, data: Data) -> None:
        probs = to_number(data[self.inputs[0]])
        if probs.shape[-1] > 1:
            probs = np.max(probs, axis=-1)
        self.max_probs.extend(probs)

    def on_epoch_end(self, data: Data) -> None:
        self.max_probs = np.array(self.max_probs)
        data.write_with_log(self.outputs[0], np.mean(self.max_probs))
        data.write_with_log(self.outputs[1], np.std(self.max_probs))
        data.write_with_log(self.outputs[2], np.median(self.max_probs))
        data.write_with_log(self.outputs[3], np.mean(self.max_probs < 0.9))


def noise_attack(x):
    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    rng = 0.1 * (mx - mn)
    return tf.clip_by_value(x + tf.random.uniform(shape=x.shape, minval=-rng, maxval=rng), mn, mx)


def eps_noise_attack(x, eps):
    return x + tf.random.uniform(shape=x.shape, minval=-eps, maxval=eps)


class Zscore(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(tf.subtract(data, mean), tf.maximum(std, 1e-8))
        return data


def build_encdec(input_shape=(32, 32, 3), latent_dim=100, epsilon=8.0 / 255):
    # encoder
    x0 = layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(x0)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    # decoder
    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    x = x0 + epsilon * x
    return Model(inputs=x0, outputs=x)


def build_encdec2(input_shape=(32, 32, 3), latent_dim=100, epsilon=8.0 / 255):
    # encoder
    img = layers.Input(shape=input_shape)
    grads = layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(grads)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    # decoder
    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    x = img + epsilon * x
    return Model(inputs=[img, grads], outputs=x)


def build_encdec3(input_shape=(32, 32, 3), latent_dim=100, epsilon=8.0 / 255):
    # encoder
    img = layers.Input(shape=input_shape)
    x0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(img)
    x0 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x0)
    x0 = tf.keras.layers.Flatten()(x0)
    x0 = tf.keras.layers.Dense(latent_dim)(x0)

    grads = layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(grads)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)

    x = tf.keras.layers.Concatenate()([x, x0])
    # decoder
    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    x = img + epsilon * x
    return Model(inputs=[img, grads], outputs=x)


def build_unet(input_shape=(32, 32, 3), epsilon=0.1) -> tf.keras.Model:
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    up_config = {'size': (2, 2), 'interpolation': 'bilinear'}
    img = Input(input_shape)
    grads = Input(input_shape)
    conv1 = Conv2D(64, 3, **conv_config)(grads)
    conv1 = Conv2D(64, 3, **conv_config)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, **conv_config)(pool1)
    conv2 = Conv2D(128, 3, **conv_config)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, **conv_config)(pool2)
    conv3 = Conv2D(256, 3, **conv_config)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, **conv_config)(pool3)
    conv4 = Conv2D(512, 3, **conv_config)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, **conv_config)(pool4)
    conv5 = Conv2D(1024, 3, **conv_config)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 3, **conv_config)(UpSampling2D(**up_config)(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, **conv_config)(merge6)
    conv6 = Conv2D(512, 3, **conv_config)(conv6)

    up7 = Conv2D(256, 3, **conv_config)(UpSampling2D(**up_config)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(256, 3, **conv_config)(merge7)
    conv7 = Conv2D(256, 3, **conv_config)(conv7)

    up8 = Conv2D(128, 3, **conv_config)(UpSampling2D(**up_config)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(128, 3, **conv_config)(merge8)
    conv8 = Conv2D(128, 3, **conv_config)(conv8)

    up9 = Conv2D(64, 3, **conv_config)(UpSampling2D(**up_config)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(64, 3, **conv_config)(merge9)
    conv9 = Conv2D(64, 3, **conv_config)(conv9)
    conv10 = Conv2D(3, 1, activation='tanh')(conv9)

    x = img + epsilon * conv10
    model = Model(inputs=[img, grads], outputs=x)
    return model


def get_estimator(epochs=160, batch_size=256, epsilon=0.1, save_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, eval_data = load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train")),
            CoarseDropout(inputs="x", outputs="x", mode="train", max_holes=1),
        ])
    # step 2: prepare network
    model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)),
                     optimizer_fn="adam")
    attacker = fe.build(model_fn=lambda: build_unet(input_shape=(32, 32, 3), epsilon=epsilon), optimizer_fn="adam")

    base_update = UpdateOp(model=model, loss_name="base_ce")
    avg_update = UpdateOp(model=model, loss_name="avg_ce")

    network = fe.Network(ops=[
        # Actual training
        Watch(inputs="x", mode=('train', 'eval', 'test')),
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        # Train the attacker
        GradientOp(inputs="x", finals="base_ce", outputs="x_g"),
        Zscore(inputs="x_g", outputs="x_g"),
        ModelOp(model=attacker, inputs=("x", "x_g"), outputs="x_gan"),
        ModelOp(model=model, inputs="x_gan", outputs="y_pred_gan"),
        CrossEntropy(inputs=("y_pred_gan", "y"), outputs="gan_ce"),
        # Adversarial attack (FGSM)
        FGSM(data="x", loss="base_ce", outputs="x_fgsm", epsilon=epsilon, mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_fgsm", outputs="y_pred_fgsm", mode=('eval', 'test')),
        CrossEntropy(inputs=("y_pred_fgsm", "y"), outputs="fgsm_ce", mode=('eval', 'test')),
        # Uniform noise attack
        LambdaOp(inputs="x", outputs="x_noise", fn=lambda x: eps_noise_attack(x, epsilon), mode=('eval', 'test')),
        ModelOp(model=model, inputs="x_noise", outputs="y_pred_u", mode=('eval', 'test')),
        # Random attack
        LambdaOp(inputs="x", outputs="rand",
                 fn=lambda x: tf.random.uniform(shape=x.shape, minval=tf.reduce_min(x), maxval=tf.reduce_max(x)),
                 mode=('eval', 'test')),
        ModelOp(model=model, inputs="rand", outputs="y_pred_rand", mode=('eval', 'test')),
        # Updates
        Average(inputs=("base_ce", "gan_ce"), outputs="avg_ce"),
        # EpochScheduler({1: base_update, 10: avg_update}),
        # avg_update,
        EpochScheduler({1: base_update, 80: None}),
        LambdaOp(inputs="gan_ce", outputs="gan_ce", fn=lambda x: -x),
        # EpochScheduler({1: None, 10: UpdateOp(model=attacker, loss_name="gan_ce")}),
        # UpdateOp(model=attacker, loss_name="gan_ce")
        EpochScheduler({1: None, 80: UpdateOp(model=attacker, loss_name="gan_ce")})
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_acc"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm", output_name="fgsm_acc"),
        Accuracy(true_key="y", pred_key="y_pred_u", output_name="+u_acc"),
        Accuracy(true_key="y", pred_key="y_pred_gan", output_name="gan_acc"),
        MCC(true_key="y", pred_key="y_pred", output_name="base_mcc"),
        MCC(true_key="y", pred_key="y_pred_fgsm", output_name="fgsm_mcc"),
        MCC(true_key="y", pred_key="y_pred_u", output_name="+u_mcc"),
        MCC(true_key="y", pred_key="y_pred_gan", output_name="gan_mcc"),
        Confidence(key="y_pred_rand"),
        BestModelSaver(model=model, save_dir=save_dir, metric="base_mcc", save_best_mode="max", load_best_final=True),
        # EarlyStopping(monitor="base_ce", patience=15),
        ReduceLROnPlateau(model=model, metric="base_ce"),
        Traceability("attack")
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=["fgsm_ce", "base_ce", "avg_ce"])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit("unet80")
    est.test()
