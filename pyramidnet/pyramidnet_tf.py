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
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=planes * 4, kernel_size=1, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    shortcut = x
    if downsample:
        shortcut = layers.AveragePooling2D()(shortcut)
    if shortcut.shape[-1] != out.shape[-1]:
        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, out.shape[-1] - shortcut.shape[-1]]])
    return out + shortcut


def make_group(x, featuremap_dim, addrate, block, block_depth, stride=1):
    for i in range(0, block_depth):
        featuremap_dim += addrate
        if i == 0:
            x = block(x=x, planes=int(round(featuremap_dim)), stride=stride, downsample=stride != 1)
        else:
            x = block(x=x, planes=int(round(featuremap_dim)), stride=1)
    return x, featuremap_dim


def pyramidnet_cifar(inputs_shape, depth, alpha, num_classes, bottleneck=False):
    if bottleneck:
        n = int((depth - 2) / 9)
        block = bottle_neck
    else:
        n = int((depth - 2) / 6)
        block = basic_block
    addrate = alpha / 3 / n
    inputs = layers.Input(shape=inputs_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x, featuremap_dim = make_group(x, 16, addrate=addrate, block=block, block_depth=n)
    x, featuremap_dim = make_group(x, featuremap_dim, addrate=addrate, block=block, block_depth=n, stride=2)
    x, featuremap_dim = make_group(x, featuremap_dim, addrate=addrate, block=block, block_depth=n, stride=2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(8)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def pyramidnet_imagenet(inputs_shape, depth, alpha, num_classes):
    assert depth in {18, 34, 50, 101, 152, 200}, "depth must be one of {}".format({18, 34, 50, 101, 152, 200})
    blocks = {18: basic_block, 34: basic_block, 50: bottle_neck, 101: bottle_neck, 152: bottle_neck, 200: bottle_neck}
    block_depths = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    addrate = alpha / sum(block_depths[depth])
    block = blocks[depth]
    block_depth = block_depths[depth]
    inputs = layers.Input(shape=inputs_shape)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x, featuremap_dim = make_group(x, 64, addrate, block=block, block_depth=block_depth[0])
    x, featuremap_dim = make_group(x, featuremap_dim, addrate, block=block, block_depth=block_depth[1], stride=2)
    x, featuremap_dim = make_group(x, featuremap_dim, addrate, block=block, block_depth=block_depth[2], stride=2)
    x, featuremap_dim = make_group(x, featuremap_dim, addrate, block=block, block_depth=block_depth[3], stride=2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(7)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    # model = pyramidnet_cifar(inputs_shape=(32, 32, 3), depth=272, alpha=200, num_classes=10, bottleneck=True)
    model = pyramidnet_imagenet(inputs_shape=(224, 224, 3), depth=200, alpha=300, num_classes=1000)
    pdb.set_trace()
