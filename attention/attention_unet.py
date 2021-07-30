import pdb
import tempfile

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, ReLU, UpSampling2D, \
    concatenate
from tensorflow.keras.models import Model

import fastestimator as fe
from fastestimator.architecture.tensorflow import UNet
from fastestimator.dataset.data import montgomery
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize, Rotate
from fastestimator.op.numpyop.univariate import Minmax, RandomGamma, ReadImage
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice


def attention_block(filter, c, x):
    c1 = Conv2D(filter, kernel_size=1)(c)
    c1 = BatchNormalization()(c1)
    x1 = Conv2D(filter, kernel_size=1)(x)
    x1 = BatchNormalization()(x1)
    att = ReLU()(x1 + c1)
    att = Conv2D(1, kernel_size=1)(att)
    att = BatchNormalization()(att)
    att = tf.sigmoid(att)
    return x * att


def attention_unet(input_size):
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
    drop4 = attention_block(512, c=up6, x=drop4)
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, **conv_config)(merge6)
    conv6 = Conv2D(512, 3, **conv_config)(conv6)

    up7 = Conv2D(256, 3, **conv_config)(UpSampling2D(**up_config)(conv6))
    conv3 = attention_block(256, c=up7, x=conv3)
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(256, 3, **conv_config)(merge7)
    conv7 = Conv2D(256, 3, **conv_config)(conv7)

    up8 = Conv2D(128, 3, **conv_config)(UpSampling2D(**up_config)(conv7))
    conv2 = attention_block(128, c=up8, x=conv2)
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(128, 3, **conv_config)(merge8)
    conv8 = Conv2D(128, 3, **conv_config)(conv8)

    up9 = Conv2D(64, 3, **conv_config)(UpSampling2D(**up_config)(conv8))
    conv1 = attention_block(64, c=up9, x=conv1)
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(64, 3, **conv_config)(merge9)
    conv9 = Conv2D(64, 3, **conv_config)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model


class CombineLeftRightMask(NumpyOp):
    def forward(self, data, state):
        mask_left, mask_right = data
        data = mask_left + mask_right
        return data


def get_estimator(epochs=40, batch_size=8, save_dir=tempfile.mkdtemp(), log_steps=5, data_dir=None):
    # step 1
    csv = montgomery.load_data(root_dir=data_dir)
    pipeline = fe.Pipeline(
        train_data=csv,
        eval_data=csv.split(0.2),
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", parent_path=csv.parent_path, outputs="image", color_flag='gray'),
            ReadImage(inputs="mask_left",
                      parent_path=csv.parent_path,
                      outputs="mask_left",
                      color_flag='gray',
                      mode='!infer'),
            ReadImage(inputs="mask_right",
                      parent_path=csv.parent_path,
                      outputs="mask_right",
                      color_flag='gray',
                      mode='!infer'),
            CombineLeftRightMask(inputs=("mask_left", "mask_right"), outputs="mask", mode='!infer'),
            Resize(image_in="image", width=256, height=256),
            Resize(image_in="mask", width=256, height=256, mode='!infer'),
            Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="mask", mode='train')),
            Sometimes(numpy_op=Rotate(
                image_in="image", mask_in="mask", limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, mode='train')),
            RandomGamma(inputs="image", outputs="image_gamma", gamma_limit=(20, 180), mode="eval"),
            Minmax(inputs="image_gamma", outputs="image_gamma", mode="eval"),
            Minmax(inputs="image", outputs="image"),
            Minmax(inputs="mask", outputs="mask", mode='!infer')
        ])

    # step 2
    model_unet = fe.build(model_fn=lambda: UNet(input_size=(256, 256, 1)),
                          optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-4),
                          model_name="unet")
    model_aunet = fe.build(model_fn=lambda: attention_unet(input_size=(256, 256, 1)),
                           optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-4),
                           model_name="aunet")

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=model_unet, outputs="pred_segment"),
        ModelOp(inputs="image_gamma", model=model_unet, outputs="pred_segment_gamma", mode="eval"),
        CrossEntropy(inputs=("pred_segment", "mask"), outputs="loss_unet", form="binary"),
        UpdateOp(model=model_unet, loss_name="loss_unet"),
        ModelOp(inputs="image", model=model_aunet, outputs="pred_segment_a"),
        ModelOp(inputs="image_gamma", model=model_aunet, outputs="pred_segment_a_gamma", mode="eval"),
        CrossEntropy(inputs=("pred_segment_a", "mask"), outputs="loss_att", form="binary"),
        UpdateOp(model=model_aunet, loss_name="loss_att")
    ])

    # step 3
    traces = [
        Dice(true_key="mask", pred_key="pred_segment_gamma", output_name="Dice_unet_gamma"),
        Dice(true_key="mask", pred_key="pred_segment_a_gamma", output_name="Dice_att_gamma"),
        Dice(true_key="mask", pred_key="pred_segment", output_name="Dice_unet_normal"),
        Dice(true_key="mask", pred_key="pred_segment_a", output_name="Dice_att_normal"),
    ]
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, log_steps=log_steps, traces=traces)
    return estimator
