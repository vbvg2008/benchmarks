import fastestimator as fe
import tensorflow as tf
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.backend import cast, squeeze
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.op import LambdaOp
from fastestimator.op.tensorop.gradient import FGSM, Watch, GradientOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp
from fastestimator.util.img_data import ImgData
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model


def attacker(input_shape=(32, 32, 3), latent_dim=100):
    # encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(latent_dim))
    # decoder
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim,)))
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


def attacker2(input_size=(32, 32, 3)) -> tf.keras.Model:
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


class Zscore(fe.op.tensorop.TensorOp):
    def forward(self, data, state):
        mean = tf.reduce_mean(data)
        std = tf.keras.backend.std(data)
        data = tf.math.divide(tf.subtract(data, mean), tf.maximum(std, 1e-8))
        return data


if __name__ == "__main__":
    batch_size = 256
    n_display = 10

    train_data, eval_data = load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data, batch_size=batch_size, ops=[
        Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        LambdaOp(inputs="y", outputs="y", fn=lambda y: cast(y, dtype='int64'))], num_process=0)

    base = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)), optimizer_fn="adam",
                    weights_path="model_best_accuracy.h5")
    enc01 = fe.build(model_fn=attacker, optimizer_fn="adam", weights_path="attacker_eps01.h5")
    enc004 = fe.build(model_fn=attacker, optimizer_fn="adam", weights_path="attacker_eps004.h5")
    unet004 = fe.build(model_fn=attacker2, optimizer_fn="adam", weights_path="attacker_eps004_unet.h5")

    batch = pipeline.get_results(mode="test")

    network = fe.Network(ops=[
        # eps01
        ModelOp(model=enc01, inputs="x", outputs="x_enc_01"),
        LambdaOp(inputs=("x", "x_enc_01"), outputs="x_enc_01", fn=lambda x, x01: x + 0.1 * x01),
        ModelOp(model=base, inputs="x_enc_01", outputs="y_enc_01"),
        # eps004
        ModelOp(model=enc004, inputs="x", outputs="x_enc_004"),
        LambdaOp(inputs=("x", "x_enc_004"), outputs="x_enc_004", fn=lambda x, x004: x + 0.04 * x004),
        ModelOp(model=base, inputs="x_enc_004", outputs="y_enc_004"),
        # clean
        Watch(inputs="x", mode=("eval", "test")),
        ModelOp(model=base, inputs="x", outputs="y_base"),
        CrossEntropy(inputs=("y_base", "y"), outputs="base_ce"),
        # fgsm01
        FGSM(data="x", loss="base_ce", outputs="x_fgsm_01", epsilon=0.1),
        ModelOp(model=base, inputs="x_fgsm_01", outputs="y_fgsm_01"),
        # fgsm004
        FGSM(data="x", loss="base_ce", outputs="x_fgsm_004", epsilon=0.04),
        ModelOp(model=base, inputs="x_fgsm_004", outputs="y_fgsm_004"),
        # unet004
        GradientOp(inputs="x", finals="base_ce", outputs="x_g"),
        Zscore(inputs="x_g", outputs="x_g"),
        ModelOp(model=unet004, inputs="x_g", outputs="x_unet_004"),
        LambdaOp(inputs=("x", "x_unet_004"), outputs="x_unet_004", fn=lambda x, x004: x + 0.04 * x004),
        ModelOp(model=base, inputs="x_unet_004", outputs="y_unet_004"),
        # Compute performances
        LambdaOp(inputs=("y", "y_base"), outputs="base_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1)),
        LambdaOp(inputs=("y", "y_fgsm_01"), outputs="fgsm_01_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1)),
        LambdaOp(inputs=("y", "y_fgsm_004"), outputs="fgsm_004_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1)),
        LambdaOp(inputs=("y", "y_enc_01"), outputs="enc_01_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1)),
        LambdaOp(inputs=("y", "y_enc_004"), outputs="enc_004_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1)),
        LambdaOp(inputs=("y", "y_unet_004"), outputs="unet_004_correct",
                 fn=lambda y, p: squeeze(y) == tf.argmax(p, axis=-1))
    ])

    processed = network.transform(batch, mode="test")

    bboxes = {key: [
        tf.where(tf.reshape(processed[key], shape=(-1, 1, 1)), tf.convert_to_tensor([[0, 0, 0, 0]]),
                 tf.convert_to_tensor([[0, 0, 31, 31]]))[:n_display],
        tf.where(tf.reshape(processed[key], shape=(-1, 1, 1)), tf.convert_to_tensor([[0, 0, 31, 31]]),
                 tf.convert_to_tensor([[0, 0, 0, 0]]))[:n_display],
    ] for key in ["base_correct", "fgsm_01_correct", "fgsm_004_correct", "enc_01_correct", "enc_004_correct",
                  "unet_004_correct"]}

    imgs = ImgData(x=[processed["x"][:n_display]] + bboxes["base_correct"],
                   x_fgsm_004=[processed["x_fgsm_004"][:n_display]] + bboxes["fgsm_004_correct"],
                   x_enc_004=[processed["x_enc_004"][:n_display]] + bboxes["enc_004_correct"],
                   x_unet_004=[processed["x_unet_004"][:n_display]] + bboxes["unet_004_correct"],
                   x_fgsm_01=[processed["x_fgsm_01"][:n_display]] + bboxes["fgsm_01_correct"],
                   x_enc_01=[processed["x_enc_01"][:n_display]] + bboxes["enc_01_correct"])
    imgs.paint_figure(save_path='attacks.png')
