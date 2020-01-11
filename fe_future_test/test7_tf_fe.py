"""test case for tensorflow users
"""
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, layers

import fastestimator as fe
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy


def Scale(dataset):
    dataset["x"] = tf.cast(dataset["x"], tf.float32)
    dataset["y"] = tf.cast(dataset["y"], tf.int32)
    dataset["x"] = dataset["x"] / 255.0
    return dataset


def get_dataset(x, y, shuffle=True):
    data = {"x": x, "y": y}
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        data_length = x.shape[0]
        ds = ds.shuffle(data_length)
    ds = ds.map(Scale, num_parallel_calls=4)
    ds = ds.batch(32)
    ds = ds.prefetch(1)
    return ds


def tf_model(input_shape=(28, 28, 1), classes=10):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def get_estimator():
    #step 1
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

    train_ds = get_dataset(x=np.expand_dims(x_train, -1), y=y_train)
    pdb.set_trace()
    pipeline = fe.Pipeline(train_data=get_dataset(x=np.expand_dims(x_train, -1), y=y_train),
                           eval_data=get_dataset(x=np.expand_dims(x_eval, -1), y=y_eval),
                           batch_size=32)
    #step 2
    model = fe.build(model_def=tf_model, optimizer_def="adam")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce")
    ])
    #step 3
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=2,
                             steps_per_epoch=1875,
                             traces=Accuracy(true_key="y", pred_key="y_pred"))
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
