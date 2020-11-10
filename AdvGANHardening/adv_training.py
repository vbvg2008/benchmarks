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
from fastestimator.op.op import LambdaOp
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def Attacker(input_shape=(32, 32, 3), latent_dim=100):
    # encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(latent_dim))
    # decoder
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


def Attacker2(input_size=(32, 32, 3)) -> tf.keras.Model:
    """A standard UNet implementation in pytorch

    Args:
        input_size: The size of the input tensor (height, width, channels).

    Returns:
        A TensorFlow LeNet model.
    """
    conv_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    up_config = {'size': (2, 2), 'interpolation': 'bilinear'}
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, **conv_config)(inputs)
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
    model = Model(inputs=inputs, outputs=conv10)
    return model


def noise_attack(x):
    mn = tf.reduce_min(x)
    mx = tf.reduce_max(x)
    rng = 0.1 * (mx - mn)
    return tf.clip_by_value(x + tf.random.uniform(shape=x.shape, minval=-rng, maxval=rng), mn, mx)


class DebugOp(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        pdb.set_trace()
        return data


class Zscore(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(tf.subtract(data, mean), tf.maximum(std, 1e-8))
        return data


def get_estimator(epochs=240, batch_size=256, save_dir=tempfile.mkdtemp(), epsilon=0.04):
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
    attack_model = fe.build(model_fn=Attacker, optimizer_fn="adam")
    target_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)),
                            optimizer_fn="adam",
                            weights_path="model_best_accuracy.h5")
    network = fe.Network(ops=[
        Watch(inputs="x", mode="eval"),
        ModelOp(model=attack_model, inputs="x", outputs="x_noise"),
        LambdaOp(fn=lambda a, b: a + epsilon * b, inputs=("x", "x_noise"), outputs="x_attack"),
        ModelOp(model=target_model, inputs="x_attack", outputs="y_pred_attack"),
        ModelOp(model=target_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred_attack", "y"), outputs="ce_attack"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss_target"),
        LambdaOp(fn=lambda x: -x, inputs="ce_attack", outputs="loss_attack"),
        # LambdaOp(fn=lambda a, b: a + b, inputs=("loss_target", "ce_attack"), outputs="loss_target", mode="train"),
        UpdateOp(model=attack_model, loss_name="loss_attack", mode="train"),
        # UpdateOp(model=target_model, loss_name="loss_target", mode="train"),
        # Adversarial attack (FGSM)
        FGSM(data="x", loss="loss_target", outputs="x_fgsm", epsilon=epsilon, mode='eval'),
        ModelOp(model=target_model, inputs="x_fgsm", outputs="y_pred_fgsm", mode='eval'),
        # Uniform noise attack
        LambdaOp(inputs="x", outputs="x_noise", fn=noise_attack, mode='eval'),
        ModelOp(model=target_model, inputs="x_noise", outputs="y_pred_u", mode='eval')
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_acc"),
        Accuracy(true_key="y", pred_key="y_pred_attack", output_name="attacker_acc"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm", output_name="fgsm_acc"),
        Accuracy(true_key="y", pred_key="y_pred_u", output_name="uni_acc")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
