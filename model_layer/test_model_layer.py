import pdb

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def model1():
    inputs = tf.keras.Input(shape=(256, 256, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling=None)
    output = backbone(inputs)
    output = layers.Dense(10)(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def model2():
    inputs = tf.keras.Input(shape=(256, 256, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs, pooling=None)
    output = backbone.layers[-1].output
    output = layers.Dense(10)(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


if __name__ == "__main__":
    model1 = model1()  # model 1 uses model as layer
    model2 = model2()  # model 2 breaks all layers
    pdb.set_trace()
