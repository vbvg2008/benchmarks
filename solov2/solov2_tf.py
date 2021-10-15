import pdb

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def fpn(C2, C3, C4, C5, l2_reg=1e-4):
    # lateral conv
    P5 = layers.Conv2D(256, kernel_size=1, kernel_regularizer=regularizers.l2(l2_reg))(C5)
    P5_up = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1, kernel_regularizer=regularizers.l2(l2_reg))(C4)
    P4 = P4 + P5_up
    P4_up = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1, kernel_regularizer=regularizers.l2(l2_reg))(C3)
    P3 = P3 + P4_up
    P3_up = layers.UpSampling2D()(P3)
    P2 = layers.Conv2D(256, kernel_size=1, kernel_regularizer=regularizers.l2(l2_reg))(C2)
    P2 = P2 + P3_up
    # fpn conv
    P5 = layers.Conv2D(256, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(P5)
    P4 = layers.Conv2D(256, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(P4)
    P3 = layers.Conv2D(256, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(P3)
    P2 = layers.Conv2D(256, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(l2_reg))(P2)
    P6 = layers.MaxPool2D()(P5)
    return P2, P3, P4, P5, P6


def solov2(input_shape, l2_reg=1e-4):
    inputs = tf.keras.Input(shape=input_shape)
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[38].name == "conv2_block3_out"
    C2 = resnet50.layers[38].output
    assert resnet50.layers[80].name == "conv3_block4_out"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "conv4_block6_out"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "conv5_block3_out"
    C5 = resnet50.layers[-1].output
    P2, P3, P4, P5, P6 = fpn(C2, C3, C4, C5, l2_reg=l2_reg)
    pdb.set_trace()


if __name__ == "__main__":

    solov2(input_shape=(1024, 1024, 3))
