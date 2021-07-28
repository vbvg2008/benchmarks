import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from util import timeit


def my_resnet50():
    inputs = layers.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    outputs = layers.Dense(1000, activation='softmax')(x)
    # outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


if __name__ == "__main__":
    model = my_resnet50()
    data = np.random.rand(1, 224, 224, 3).astype("float32")
    timeit(f=lambda: single_inference(model=model, data=data))
