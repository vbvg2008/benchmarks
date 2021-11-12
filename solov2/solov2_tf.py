import pdb

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from fastestimator.dataset.data import mscoco
from tensorflow.keras import layers


def fpn(C2, C3, C4, C5):
    # lateral conv
    P5 = layers.Conv2D(256, kernel_size=1)(C5)
    P5_up = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1)(C4)
    P4 = P4 + P5_up
    P4_up = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1)(C3)
    P3 = P3 + P4_up
    P3_up = layers.UpSampling2D()(P3)
    P2 = layers.Conv2D(256, kernel_size=1)(C2)
    P2 = P2 + P3_up
    # fpn conv
    P5 = layers.Conv2D(256, kernel_size=3, padding="same")(P5)
    P4 = layers.Conv2D(256, kernel_size=3, padding="same")(P4)
    P3 = layers.Conv2D(256, kernel_size=3, padding="same")(P3)
    P2 = layers.Conv2D(256, kernel_size=3, padding="same")(P2)
    return P2, P3, P4, P5


def pad_with_coord(inputs):
    batch_size, height, width, _ = tf.shape(inputs)
    x_ranges = tf.cast(tf.linspace(-1, 1, num=width), inputs.dtype)
    y_ranges = tf.cast(tf.linspace(-1, 1, num=height), inputs.dtype)
    X, Y = tf.meshgrid(x_ranges, y_ranges)
    X = tf.tile(X[tf.newaxis, ..., tf.newaxis], [batch_size, 1, 1, 1])
    Y = tf.tile(Y[tf.newaxis, ..., tf.newaxis], [batch_size, 1, 1, 1])
    outputs = tf.concat([inputs, X, Y], axis=-1)
    return outputs


def conv_norm(x, filters, kernel_size=3, groups=32):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = tfa.layers.GroupNormalization(groups=groups, epsilon=1e-5)(x)
    return x


def solov2_head_model(stacked_convs=4, ch_in=258, ch_feature=512, ch_kernel_out=256, num_classes=80):
    inputs = layers.Input(shape=(None, None, ch_in))
    feature_kernel = inputs
    feature_cls = inputs[..., :-2]
    for _ in range(stacked_convs):
        feature_kernel = tf.nn.relu(conv_norm(feature_kernel, filters=ch_feature))
        feature_cls = tf.nn.relu(conv_norm(feature_cls, filters=ch_feature))
    feature_kernel = layers.Conv2D(filters=ch_kernel_out,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01))(feature_kernel)
    feature_cls = layers.Conv2D(filters=num_classes,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.initializers.constant(np.log(1 / 99)))(feature_cls)
    return tf.keras.Model(inputs=inputs, outputs=[feature_kernel, feature_cls])


def solov2_head(P2, P3, P4, P5, num_classes=80):
    head_model = solov2_head_model(num_classes=num_classes)
    # applying maxpool first for P2
    P2 = layers.MaxPool2D()(P2)
    features = [P2, P3, P4, P5, P5]
    grid_sizes = [40, 36, 24, 16, 12]
    feat_kernel_list, feat_cls_list = [], []
    for feature, grid_size in zip(features, grid_sizes):
        feature = pad_with_coord(feature)
        feature = tf.image.resize(feature, size=(grid_size, grid_size))
        feat_kernel, feat_cls = head_model(feature)
        feat_kernel_list.append(feat_kernel)
        feat_cls_list.append(feat_cls)
    return feat_cls_list, feat_kernel_list


def solov2_maskhead(P2, P3, P4, P5, mid_ch=128, out_ch=256):
    # first level
    P2 = tf.nn.relu(conv_norm(P2, filters=mid_ch))
    # second level
    P3 = tf.nn.relu(conv_norm(P3, filters=mid_ch))
    P3 = layers.UpSampling2D()(P3)
    # third level
    P4 = tf.nn.relu(conv_norm(P4, filters=mid_ch))
    P4 = layers.UpSampling2D()(P4)
    P4 = tf.nn.relu(conv_norm(P4, filters=mid_ch))
    P4 = layers.UpSampling2D()(P4)
    # top level, add coordinate
    P5 = pad_with_coord(P5)
    P5 = tf.nn.relu(conv_norm(P5, filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    P5 = tf.nn.relu(conv_norm(P5, filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    P5 = tf.nn.relu(conv_norm(P5, filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    seg_outputs = tf.nn.relu(conv_norm(P2 + P3 + P4 + P5, filters=out_ch, kernel_size=1))
    return seg_outputs


def solov2(input_shape=(None, None, 3), num_classes=80):
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
    P2, P3, P4, P5 = fpn(C2, C3, C4, C5)
    feat_seg = solov2_maskhead(P2, P3, P4, P5)  # [B, h/4, w/4, 256]
    feat_cls_list, feat_kernel_list = solov2_head(P2, P3, P4, P5, num_classes=num_classes)  # [B, grid, grid, 80], [B, grid, grid, 256]
    model = tf.keras.Model(inputs=inputs, outputs=[feat_seg, feat_cls_list, feat_kernel_list])
    return model


def get_estimator(data_dir):
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir, load_masks=True)
    pipeline = fe.Pipeline()
    pdb.set_trace()


if __name__ == "__main__":
    model = solov2(input_shape=(1024, 1024, 3))
    x = tf.ones([2, 1024, 1024, 3])
    feat_seg, feat_cls_list, feat_kernel_list = model(x)
    pdb.set_trace()
