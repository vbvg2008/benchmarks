from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from util import timeit


def ResNet9(input_size: Tuple[int, int, int] = (32, 32, 3), classes: int = 10) -> tf.keras.Model:
    # prep layers
    inp = layers.Input(shape=input_size)
    x = layers.Conv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer1
    x = layers.Conv2D(128, 3, padding='same', groups=2)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    # layer2
    x = layers.Conv2D(256, 3, padding='same', groups=2)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # layer3
    x = layers.Conv2D(512, 3, padding='same', groups=2)(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    # layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
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
    x = layers.Conv2D(num_channel, 3, padding='same', groups=2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(num_channel, 3, padding='same', groups=2)(x)
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
