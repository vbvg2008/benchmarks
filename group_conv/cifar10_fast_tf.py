import tempfile

import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop import CoarseDropout, HorizontalFlip, Minmax, Normalize, Onehot, PadIfNeeded, \
    RandomBrightnessContrast, RandomCrop, Sometimes
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


class NormalizedConv2D(layers.Conv2D):
    def call(self, x):
        pixel_sum = tf.nn.conv2d(x,
                                 tf.ones(list(self.kernel_size) + [x.shape[-1], self.filters]),
                                 strides=self.strides,
                                 padding=self.padding.upper())
        conv_result = super().call(x)
        result = (conv_result + conv_result / (tf.abs(pixel_sum) + 0.1)) / 2
        return result


def residual(x, num_channel):
    x = NormalizedConv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = NormalizedConv2D(num_channel, 3, padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def my_model():
    #prep layers
    inp = layers.Input(shape=(32, 32, 3))
    x = NormalizedConv2D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer1
    x = NormalizedConv2D(128, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 128)])
    #layer2
    x = NormalizedConv2D(256, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #layer3
    x = NormalizedConv2D(512, 3, padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Add()([x, residual(x, 512)])
    #layers4
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    return model


def get_estimator(epochs=10, batch_size=512, save_dir=tempfile.mkdtemp()):
    # step 1: prepare dataset
    train_data, eval_data = load_data()
    test_data = eval_data.split(0.5)
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        batch_size=batch_size,
        ops=[
            RandomBrightnessContrast(inputs="x", outputs="x", mode="test", brightness_limit=0.4, contrast_limit=0.4),
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        ])

    # step 2: prepare network
    model = fe.build(model_fn=my_model, optimizer_fn="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda step: cosine_decay(step, cycle_length=2352, init_lr=1e-3))
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)

    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
    est.test()
