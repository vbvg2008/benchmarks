import pdb

import tensorflow as tf
from tensorflow.keras import layers


def basic_block(x, planes, stride=1, downsample=False):
    out = layers.BatchNormalization()(x)
    out = layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=planes, kernel_size=3, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    shortcut = x
    if downsample:
        shortcut = layers.AveragePooling2D()(shortcut)
    if shortcut.shape[-1] != out.shape[-1]:
        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, out.shape[-1] - shortcut.shape[-1]]])
    return out + shortcut

def bottle_neck(x, planes, stride=1, downsample=False):
    out = layers.BatchNormalization()(x)
    out = layers.Conv2D(filters=planes, kernel_size=1, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(filters=planes * 4, kernel_size=1, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)



def pyramid_net(dataset, depth, alpha, num_classes, bottleneck=False):
    assert dataset in {"cifar10", "cifar100", "imagenet"}, "dataset must be one of 'cifar10', 'cifar100' or 'imagenet'"
    assert depth in {18, 34, 50, 101, 152, 200}, "depth must be one of {}".format({18, 34, 50, 101, 152, 200})


if __name__ == "__main__":
    inputs = layers.Input(shape=(224, 224, 3))
    output = basic_block(inputs, planes=32)
    pdb.set_trace()
