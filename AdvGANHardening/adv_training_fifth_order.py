import pdb
import tempfile

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.tensorop import Average, LambdaOp
from fastestimator.op.tensorop.gradient import FGSM, GradientOp, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def Attacker(input_shape=(32, 32, 3), epsilon=0.1):
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


def noise_attack(x):
    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    rng = 0.1 * (mx - mn)
    return tf.clip_by_value(x + tf.random.uniform(shape=x.shape, minval=-rng, maxval=rng), mn, mx)


class Zscore(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(tf.subtract(data, mean), tf.maximum(std, 1e-8))
        return data


def get_estimator(epochs=500, batch_size=256, save_dir=tempfile.mkdtemp(), epsilon=0.1):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=test_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train"))
        ])
    # step 2: prepare network
    attack_model = fe.build(model_fn=lambda: Attacker(input_shape=(32, 32, 3), epsilon=epsilon / 5),
                            optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    target_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)),
                            optimizer_fn="adam",
                            weights_path="model_best_accuracy.h5")
    network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=target_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="base_ce"),
        # attack 1
        GradientOp(inputs="x", finals="base_ce", outputs="x_g"),
        Zscore(inputs="x_g", outputs="x_g"),
        ModelOp(model=attack_model, inputs=("x", "x_g"), outputs="x_a1"),
        ModelOp(model=target_model, inputs="x_a1", outputs="y_pred_a1"),
        CrossEntropy(inputs=("y_pred_a1", "y"), outputs="ce_a1"),
        # attack 2
        GradientOp(inputs="x_a1", finals="ce_a1", outputs="x_g1"),
        Zscore(inputs="x_g1", outputs="x_g1"),
        ModelOp(model=attack_model, inputs=("x_a1", "x_g1"), outputs="x_a2"),
        ModelOp(model=target_model, inputs="x_a2", outputs="y_pred_a2"),
        CrossEntropy(inputs=("y_pred_a2", "y"), outputs="ce_a2"),
        # attack 3
        GradientOp(inputs="x_a2", finals="ce_a2", outputs="x_g2"),
        Zscore(inputs="x_g2", outputs="x_g2"),
        ModelOp(model=attack_model, inputs=("x_a2", "x_g2"), outputs="x_a3"),
        ModelOp(model=target_model, inputs="x_a3", outputs="y_pred_a3"),
        CrossEntropy(inputs=("y_pred_a3", "y"), outputs="ce_a3"),
        # attack 4
        GradientOp(inputs="x_a3", finals="ce_a3", outputs="x_g3"),
        Zscore(inputs="x_g3", outputs="x_g3"),
        ModelOp(model=attack_model, inputs=("x_a3", "x_g3"), outputs="x_a4"),
        ModelOp(model=target_model, inputs="x_a4", outputs="y_pred_a4"),
        CrossEntropy(inputs=("y_pred_a4", "y"), outputs="ce_a4"),
        # attack 5
        GradientOp(inputs="x_a4", finals="ce_a4", outputs="x_g4"),
        Zscore(inputs="x_g4", outputs="x_g4"),
        ModelOp(model=attack_model, inputs=("x_a4", "x_g4"), outputs="x_a5"),
        ModelOp(model=target_model, inputs="x_a5", outputs="y_pred_a5"),
        CrossEntropy(inputs=("y_pred_a5", "y"), outputs="ce_a5"),
        # update
        Average(inputs=("ce_a1", "ce_a2", "ce_a3", "ce_a4", "ce_a5"), outputs="gan_ce"),
        LambdaOp(fn=lambda x: -x, inputs="gan_ce", outputs="gan_ce"),
        UpdateOp(model=attack_model, loss_name="gan_ce"),
        Average(inputs=("base_ce", "ce_a1", "ce_a2", "ce_a3", "ce_a4", "ce_a5"), outputs="avg_ce"),
        UpdateOp(model=target_model, loss_name="avg_ce"),
        # Adversarial attack (FGSM)
        FGSM(data="x", loss="base_ce", outputs="x_fgsm", epsilon=epsilon, mode='eval'),
        ModelOp(model=target_model, inputs="x_fgsm", outputs="y_pred_fgsm", mode='eval'),
        # Uniform noise attack
        LambdaOp(inputs="x", outputs="x_noise", fn=noise_attack, mode='eval'),
        ModelOp(model=target_model, inputs="x_noise", outputs="y_pred_u", mode='eval')
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_acc"),
        Accuracy(true_key="y", pred_key="y_pred_a1", output_name="a1_acc"),
        Accuracy(true_key="y", pred_key="y_pred_a2", output_name="a2_acc"),
        Accuracy(true_key="y", pred_key="y_pred_a3", output_name="a3_acc"),
        Accuracy(true_key="y", pred_key="y_pred_a4", output_name="a4_acc"),
        Accuracy(true_key="y", pred_key="y_pred_a5", output_name="a5_acc"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm", output_name="fgsm_acc"),
        Accuracy(true_key="y", pred_key="y_pred_u", output_name="uni_acc")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
