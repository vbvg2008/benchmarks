import pdb
import tempfile

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.models import Model

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet
from fastestimator.dataset.data.cifar10 import load_data
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, PadIfNeeded, RandomCrop
from fastestimator.op.numpyop.univariate import Normalize
from fastestimator.op.op import LambdaOp
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.gradient import FGSM, Watch
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy


def get_estimator(epochs=53, batch_size=256, save_dir=tempfile.mkdtemp(), epsilon=0.04):
    # step 1: prepare dataset
    train_data, eval_data = load_data()
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x", mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(min_height=40, min_width=40, image_in="x", image_out="x", mode="train"),
            RandomCrop(32, 32, image_in="x", image_out="x", mode="train"),
            Sometimes(HorizontalFlip(image_in="x", image_out="x", mode="train"))
        ])

    # step 2: prepare network
    target_model = fe.build(model_fn=lambda: LeNet(input_shape=(32, 32, 3)),
                            optimizer_fn="adam",
                            weights_path="../AdvGANHardening/model_best_accuracy.h5")

    network = fe.Network(ops=[
        Watch(inputs="x"),
        ModelOp(model=target_model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="loss"),
        FGSM(data="x", loss="loss", outputs="x_attack1", epsilon=epsilon / 5),
        ModelOp(model=target_model, inputs="x_attack1", outputs="y_pred_fgsm1"),
        CrossEntropy(inputs=("y_pred_fgsm1", "y"), outputs="loss_attack1"),
        FGSM(data="x_attack1", loss="loss_attack1", outputs="x_attack2", epsilon=epsilon / 5),
        ModelOp(model=target_model, inputs="x_attack2", outputs="y_pred_fgsm2"),
        CrossEntropy(inputs=("y_pred_fgsm2", "y"), outputs="loss_attack2"),
        FGSM(data="x_attack2", loss="loss_attack2", outputs="x_attack3", epsilon=epsilon / 5),
        ModelOp(model=target_model, inputs="x_attack3", outputs="y_pred_fgsm3"),
        CrossEntropy(inputs=("y_pred_fgsm3", "y"), outputs="loss_attack3"),
        FGSM(data="x_attack3", loss="loss_attack3", outputs="x_attack4", epsilon=epsilon / 5),
        ModelOp(model=target_model, inputs="x_attack4", outputs="y_pred_fgsm4"),
        CrossEntropy(inputs=("y_pred_fgsm4", "y"), outputs="loss_attack4"),
        FGSM(data="x_attack4", loss="loss_attack4", outputs="x_attack5", epsilon=epsilon / 5),
        ModelOp(model=target_model, inputs="x_attack5", outputs="y_pred_fgsm5"),
        CrossEntropy(inputs=("y_pred_fgsm5", "y"), outputs="loss_attack5"),
        Average(inputs=("loss", "loss_attack1", "loss_attack2", "loss_attack3", "loss_attack4", "loss_attack5"),
                outputs="loss_avg"),
        UpdateOp(model=target_model, loss_name="loss_avg")
    ])

    # step 3: prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name="acc"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm1", output_name="acc_fgsm1"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm2", output_name="acc_fgsm2"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm3", output_name="acc_fgsm3"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm4", output_name="acc_fgsm4"),
        Accuracy(true_key="y", pred_key="y_pred_fgsm5", output_name="acc_fgsm5")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
