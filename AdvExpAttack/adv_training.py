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


def Attacker(input_shape=(32, 32, 3), num_classes=10):
    # encoder
    image = layers.Input(shape=input_shape)
    x = tf.keras.layers.InputLayer(input_shape=input_shape)(image)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    cls_output = layers.Dense(num_classes, activation='softmax')(x)
    # decoder
    x = layers.Dense(8 * 8 * 256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x_noise = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x)
    model = Model(inputs=image, outputs=[x_noise, cls_output])
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


def get_estimator(epochs=53, batch_size=256, save_dir=tempfile.mkdtemp(), epsilon=0.04):
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
                            weights_path="../AdvGANHardening/model_best_accuracy.h5")
    network = fe.Network(ops=[
        Watch(inputs="x", mode="eval"),
        ModelOp(model=attack_model, inputs="x", outputs=["x_perturb", "attack_y_pred"]),
        LambdaOp(fn=lambda a, b: a + epsilon * b, inputs=("x", "x_perturb"), outputs="x_attack"),
        ModelOp(model=target_model, inputs="x_attack", outputs="y_pred_attack"),
        ModelOp(model=target_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred_attack", "y"), outputs="ce_attack"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss_target"),
        CrossEntropy(inputs=("attack_y_pred", "y"), outputs="attack_ce"),
        LambdaOp(fn=lambda a, b: a - b, inputs=("attack_ce", "ce_attack"), outputs="loss_attack"),
        # LambdaOp(fn=lambda a, b: a + b, inputs=("loss_target", "ce_attack"), outputs="loss_target", mode="train"),
        UpdateOp(model=attack_model, loss_name="loss_attack", mode="train"),
        # UpdateOp(model=target_model, loss_name="loss_target", mode="train"),
        # Adversarial attack (FGSM)
        FGSM(data="x", loss="attack_ce", outputs="x_fgsm", epsilon=epsilon, mode='eval'),
        ModelOp(model=attack_model, inputs="x_fgsm", outputs=("dummy", "y_pred_fgsm"), mode='eval'),
        # Uniform noise attack
        LambdaOp(inputs="x", outputs="x_noise", fn=noise_attack, mode='eval'),
        ModelOp(model=attack_model, inputs="x_noise", outputs=("dummy", "y_pred_u"), mode='eval')
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="base_acc"),
        Accuracy(true_key="y", pred_key="attack_y_pred", output_name="attack_model_acc"),
        Accuracy(true_key="y", pred_key="y_pred_attack", output_name="attack_on_target_acc"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm", output_name="fgsm_acc"),
        Accuracy(true_key="y", pred_key="y_pred_u", output_name="uni_acc")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
