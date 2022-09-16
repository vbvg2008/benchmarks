from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from util import timeit


def ResNet9(input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 10) -> tf.keras.Model:
    # prep layers
    inp = layers.Input(shape=input_size)
    x_1 = layers.Conv2D(32, 3, padding='same')(inp)
    x_2 = layers.Conv2D(32, 3, padding='same')(inp)


    x_1 = layers.BatchNormalization(momentum=0.8)(x_1)
    x_1 = layers.LeakyReLU(alpha=0.1)(x_1)
    x_2 = layers.BatchNormalization(momentum=0.8)(x_2)
    x_2 = layers.LeakyReLU(alpha=0.1)(x_2)
    # layer1
    x_1 = layers.Conv2D(64, 3, padding='same')(x_1)
    x_1 = layers.MaxPool2D()(x_1)
    x_1 = layers.BatchNormalization(momentum=0.8)(x_1)
    x_1 = layers.LeakyReLU(alpha=0.1)(x_1)
    x_1 = layers.Add()([x_1, residual(x_1, 64)])

    x_2 = layers.Conv2D(64, 3, padding='same')(x_2)
    x_2 = layers.MaxPool2D()(x_2)
    x_2 = layers.BatchNormalization(momentum=0.8)(x_2)
    x_2 = layers.LeakyReLU(alpha=0.1)(x_2)
    x_2 = layers.Add()([x_2, residual(x_2, 64)])
    # layer2
    x_1 = layers.Conv2D(128, 3, padding='same')(x_1)
    x_1 = layers.MaxPool2D()(x_1)
    x_1 = layers.BatchNormalization(momentum=0.8)(x_1)
    x_1 = layers.LeakyReLU(alpha=0.1)(x_1)

    x_2 = layers.Conv2D(128, 3, padding='same')(x_2)
    x_2 = layers.MaxPool2D()(x_2)
    x_2 = layers.BatchNormalization(momentum=0.8)(x_2)
    x_2 = layers.LeakyReLU(alpha=0.1)(x_2)
    # layer3
    x_1 = layers.Conv2D(256, 3, padding='same')(x_1)
    x_1 = layers.MaxPool2D()(x_1)
    x_1 = layers.BatchNormalization(momentum=0.8)(x_1)
    x_1 = layers.LeakyReLU(alpha=0.1)(x_1)
    x_1 = layers.Add()([x_1, residual(x_1, 256)])

    x_2 = layers.Conv2D(256, 3, padding='same')(x_2)
    x_2 = layers.MaxPool2D()(x_2)
    x_2 = layers.BatchNormalization(momentum=0.8)(x_2)
    x_2 = layers.LeakyReLU(alpha=0.1)(x_2)
    x_2 = layers.Add()([x_2, residual(x_2, 256)])
    # layers4
    x_1 = layers.GlobalMaxPool2D()(x_1)
    x_1 = layers.Flatten()(x_1)

    x_2 = layers.GlobalMaxPool2D()(x_2)
    x_2 = layers.Flatten()(x_2)

    x = layers.concatenate([x_1, x_2], axis=-1)

    x = layers.Dense(classes)(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


def residual(x: tf.Tensor, num_channel: int) -> tf.Tensor:
    """A ResNet unit for ResNet9.
    Args:
        x: Input Keras tensor.
        num_channel: The number of layer channel.
    Return:
        Output Keras tensor.
    """
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


if __name__ == "__main__":
    model = ResNet9(input_size=(1024, 1024, 3), classes=100)
    data = np.random.rand(1, 1024, 1024, 3).astype("float32")
    timeit(f=lambda: single_inference(model=model, data=data), num_runs=500)
