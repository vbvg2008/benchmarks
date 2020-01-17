import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, layers

import fastestimator as fe
from fastestimator.op.tensorop import Gradients, ModelOp, Scale, SparseCategoricalCrossentropy, UpdateOp
from fastestimator.trace import Accuracy, ModelSaver


def ResNet101(input_shape=(32, 32, 3), classes=10):
    """Creates a LeNet model with 3 convolutional and 2 dense layers.

    Args:
        input_shape (tuple, optional): shape of the input data. Defaults to (32, 32, 3).
        classes (int, optional): number of classes. Defaults to 10.

    Returns:
        'Model' object: LeNet model.
    """
    inputs = layers.Input(shape=input_shape)
    resnet101 = tf.keras.applications.ResNet101(weights=None, include_top=False, input_tensor=inputs, pooling="avg")
    x = resnet101.outputs[0]
    x = layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def get_estimator(epochs=30, batch_size=192, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    train_data = {"x": x_train, "y": y_train}
    eval_data = {"x": x_eval, "y": y_eval}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Scale(scalar=1.0 / 255, inputs="x", outputs="x"))

    # step 2. prepare model
    model = fe.build(model_def=ResNet101, model_name="resnet101", optimizer=tf.optimizers.Adam(0.001), loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss"),
        Gradients(loss="loss", models=model, outputs="gradients"),
        UpdateOp(model=model, gradients="gradients")
    ])

    # step 3.prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
        ModelSaver(model_name="resnet101", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
