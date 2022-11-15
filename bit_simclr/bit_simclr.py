import tempfile

import tensorflow as tf
from models import ResnetV2

import fastestimator as fe
from fastestimator.dataset.data.cifair10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import ColorJitter, GaussianBlur, ToFloat, ToGray
from fastestimator.op.tensorop import LambdaOp, TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver
from fastestimator.trace.metric import Accuracy


class NTXentOp(TensorOp):
    def __init__(self, arg1, arg2, outputs, temperature=1.0, mode=None):
        super().__init__(inputs=(arg1, arg2), outputs=outputs, mode=mode)
        self.temperature = temperature

    def forward(self, data, state):
        arg1, arg2 = data
        loss = NTXent(arg1, arg2, self.temperature)
        return loss


def NTXent(A, B, temperature):
    large_number = 1e9
    batch_size = tf.shape(A)[0]
    A = tf.math.l2_normalize(A, -1)
    B = tf.math.l2_normalize(B, -1)

    mask = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), 2 * batch_size)

    aa = tf.matmul(A, A, transpose_b=True) / temperature
    aa = aa - mask * large_number
    ab = tf.matmul(A, B, transpose_b=True) / temperature
    bb = tf.matmul(B, B, transpose_b=True) / temperature
    bb = bb - mask * large_number
    ba = tf.matmul(B, A, transpose_b=True) / temperature
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([ab, aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([ba, bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss, ab, labels


def bit_model():
    model = ResnetV2(num_units=(3, 4, 6, 3),
                     num_outputs=128,
                     filters_factor=4,
                     trainable=True,
                     dtype=tf.float32)

    model.build((None, None, None, 3))
    return model


def pretrain_model(epochs, batch_size, train_steps_per_epoch, save_dir):
    # step 1: prepare dataset
    train_data, test_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        batch_size=batch_size,
        ops=[
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x"),

            # augmentation 1
            RandomCrop(32, 32, image_in="x", image_out="x_aug"),
            Sometimes(HorizontalFlip(image_in="x_aug", image_out="x_aug"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug", outputs="x_aug", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug", outputs="x_aug"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug", outputs="x_aug", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ToFloat(inputs="x_aug", outputs="x_aug"),

            # augmentation 2
            RandomCrop(32, 32, image_in="x", image_out="x_aug2"),
            Sometimes(HorizontalFlip(image_in="x_aug2", image_out="x_aug2"), prob=0.5),
            Sometimes(
                ColorJitter(inputs="x_aug2", outputs="x_aug2", brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                prob=0.8),
            Sometimes(ToGray(inputs="x_aug2", outputs="x_aug2"), prob=0.2),
            Sometimes(GaussianBlur(inputs="x_aug2", outputs="x_aug2", blur_limit=(3, 3), sigma_limit=(0.1, 2.0)),
                      prob=0.5),
            ToFloat(inputs="x_aug2", outputs="x_aug2")
        ])

    # step 2: prepare network
    model_con = fe.build(model_fn=bit_model, optimizer_fn="adam")
    network = fe.Network(ops=[
        LambdaOp(lambda x, y: tf.concat([x, y], axis=0), inputs=["x_aug", "x_aug2"], outputs="x_com"),
        ModelOp(model=model_con, inputs="x_com", outputs="y_com"),
        LambdaOp(lambda x: tf.split(x, 2, axis=0), inputs="y_com", outputs=["y_pred", "y_pred2"]),
        NTXentOp(arg1="y_pred", arg2="y_pred2", outputs=["NTXent", "logit", "label"]),
        UpdateOp(model=model_con, loss_name="NTXent")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="label", pred_key="logit", mode="train", output_name="contrastive_accuracy"),
        ModelSaver(model=model_con, save_dir=save_dir),
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=train_steps_per_epoch,
                             log_steps=10)
    estimator.fit()


def fastestimator_run(epochs_pretrain=50, batch_size=512, train_steps_per_epoch=None, save_dir=tempfile.mkdtemp()):

    pretrain_model(epochs_pretrain, batch_size, train_steps_per_epoch, save_dir)


if __name__ == "__main__":
    fastestimator_run()
