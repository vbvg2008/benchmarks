import pdb

import tensorflow as tf
from tensorflow.keras import layers


def conv_block(x, c, k=1, s=1):
    x = layers.Conv2D(filters=c, kernel_size=k, strides=s, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.97)(x)
    x = tf.nn.silu(x)
    return x


def bottleneck(x, c, k=1, shortcut=True):
    out = conv_block(x, c=c, k=1)
    out = conv_block(out, c=c, k=3)
    if shortcut and c == x.shape[-1]:
        out = out + x
    return out


def csp_bottleneck_conv3(x, c, n=1, shortcut=True):
    out1 = conv_block(x, c=c // 2)
    for _ in range(n):
        out1 = bottleneck(out1, c=c // 2, shortcut=shortcut)
    out2 = conv_block(x, c=c // 2)
    out = tf.concat([out1, out2], axis=-1)
    out = conv_block(out, c=c)
    return out


def spatial_pyramid_pooling(x, c, k=(5, 9, 13)):
    input_c = x.shape[-1]
    x = conv_block(x, c=input_c // 2)
    x = tf.concat([x] + [layers.MaxPool2D(pool_size=p, strides=1, padding='same')(x) for p in k], axis=-1)
    x = conv_block(x, c=c)
    return x


def yolov5(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = tf.concat([inp[:, ::2, ::2, :], inp[:, 1::2, ::2, :], inp[:, ::2, 1::2, :], inp[:, 1::2, 1::2, :]], axis=-1)
    x = conv_block(x, c=32, k=3)
    x = conv_block(x, c=64, k=3, s=2)
    x = csp_bottleneck_conv3(x, c=64)
    x = conv_block(x, c=128, k=3, s=2)
    x_4 = csp_bottleneck_conv3(x, c=128, n=3)
    x = conv_block(x_4, c=256, k=3, s=2)
    x_6 = csp_bottleneck_conv3(x, c=256, n=3)
    x = conv_block(x_6, c=512, k=3, s=2)
    x = spatial_pyramid_pooling(x, c=512)
    x = csp_bottleneck_conv3(x, 512, shortcut=False)
    x_10 = conv_block(x, 256)
    x = layers.UpSampling2D()(x_10)
    x = tf.concat([x, x_6], axis=-1)
    x = csp_bottleneck_conv3(x, 256, shortcut=False)
    x_14 = conv_block(x, 128)
    x = layers.UpSampling2D()(x_14)
    x = tf.concat([x, x_4], axis=-1)
    x_17 = csp_bottleneck_conv3(x, 128, shortcut=False)
    x = conv_block(x_17, 128, 3, 2)
    x = tf.concat([x, x_14], axis=-1)
    x_20 = csp_bottleneck_conv3(x, 256, shortcut=False)
    x = conv_block(x_20, 256, 3, 2)
    x = tf.concat([x, x_10], axis=-1)
    x_23 = csp_bottleneck_conv3(x, 512, shortcut=False)
    out_17 = layers.Conv2D((num_classes + 4) * 3, 1)(x_17)
    out_17 = layers.Reshape((-1, num_classes + 4))(out_17)
    out_20 = layers.Conv2D((num_classes + 4) * 3, 1)(x_20)
    out_20 = layers.Reshape((-1, num_classes + 4))(out_20)
    out_23 = layers.Conv2D((num_classes + 4) * 3, 1)(x_23)
    out_23 = layers.Reshape((-1, num_classes + 4))(out_23)
    results = tf.concat([out_17, out_20, out_23], axis=-2)
    return tf.keras.Model(inputs=inp, outputs=results)


if __name__ == "__main__":
    model = yolov5(input_shape=(1280, 1280, 3), num_classes=90)
    pdb.set_trace()
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # print(layer)
            print(layer.count_params())
