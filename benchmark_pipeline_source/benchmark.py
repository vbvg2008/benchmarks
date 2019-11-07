import os
import time

import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture import LeNet
from fastestimator.dataset.mnist import load_data
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy


def MyGen(x, y):
    while True:
        num_data = y.shape[0]
        idx = np.random.randint(0, num_data)
        yield {"x": x[idx], "y": y[idx]}


def get_in_memory_pipeline(batch_size):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
    eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Minmax(inputs="x", outputs="x"))
    return pipeline


def get_generator_pipeline(batch_size):
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)
    data = {"train": lambda: MyGen(x_train, y_train), "eval": lambda: MyGen(x_eval, y_eval)}
    pipeline = fe.Pipeline(data=data, batch_size=batch_size, ops=Minmax(inputs="x", outputs="x"))
    return pipeline


def get_disk_pipeline(batch_size):
    train_csv, eval_csv, path = load_data()
    # for writer, we can create two unpaired dataset (see tutorial 10)
    writer = fe.RecordWriter(save_dir=os.path.join(path, "tfrecord"),
                             train_data=train_csv,
                             validation_data=eval_csv,
                             ops=ImageReader(inputs="x", outputs="x", parent_path=path))
    pipeline = fe.Pipeline(data=writer, batch_size=batch_size, ops=Minmax(inputs="x", outputs="x"))
    return pipeline


def get_estimator(pipeline_option, batch_size=32):
    pipeline_map = {
        "tensorslice": get_in_memory_pipeline, "generator": get_generator_pipeline, "tfrecord": get_disk_pipeline
    }
    pipeline = pipeline_map[pipeline_option](batch_size)
    # step 2. prepare model
    model = fe.build(model_def=LeNet, model_name="lenet", optimizer="adam", loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss")
    ])

    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=2, steps_per_epoch=1700, validation_steps=300)
    return estimator


if __name__ == "__main__":
    batch_size = 32
    print("benchmarking in memory tensor slice pipeline:")
    pipeline_memory = get_in_memory_pipeline(batch_size)
    pipeline_memory.benchmark()
    print("benchmarking in in memory generator pipeline:")
    pipeline_generator = get_generator_pipeline(batch_size)
    pipeline_generator.benchmark()
    print("benchmarking in disk tfrecord pipeline:")
    pipeline_disk = get_disk_pipeline(batch_size)
    pipeline_disk.benchmark()
    print("benchmarking pure generator:")
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)
    original_gen = MyGen(x_train, y_train)
    tic = time.time()
    for idx in range(1000):
        data = next(original_gen)
        if idx > 0 and idx % 100 == 0:
            elapse = time.time() - tic
            example_sec = 100 / elapse
            print("step: {}, image/sec: {}".format(idx, example_sec))
            tic = time.time()
