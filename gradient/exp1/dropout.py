import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, layers

import fastestimator as fe
from fastestimator.op.tensorop import Gradients, Minmax, ModelOp, SparseCategoricalCrossentropy, UpdateOp
from fastestimator.trace import Accuracy, ModelSaver


def LeNet(input_shape=(28, 28, 1), classes=10):
    """Creates a LeNet model with 3 convolutional and 2 dense layers.

    Args:
        input_shape (tuple, optional): shape of the input data. Defaults to (28, 28, 1).
        classes (int, optional): number of classes. Defaults to 10.

    Returns:
        'Model' object: LeNet model.
    """
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def get_estimator(epochs=30, batch_size=64, steps_per_epoch=None, validation_steps=None, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.fashion_mnist.load_data()
    train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
    eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))

    # step 2. prepare model
    model = fe.build(model_def=LeNet,
                     model_name="lenet",
                     optimizer=tf.optimizers.SGD(learning_rate=0.001),
                     loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss"),
        Gradients(loss="loss", models=model, outputs="gradients"),
        UpdateOp(model=model, gradients="gradients")
    ])

    # step 3.prepare estimator
    traces = [
        Accuracy(true_key="y", pred_key="y_pred", output_name='acc'),
        ModelSaver(model_name="lenet", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(network=network,
                             pipeline=pipeline,
                             epochs=epochs,
                             traces=traces,
                             steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
