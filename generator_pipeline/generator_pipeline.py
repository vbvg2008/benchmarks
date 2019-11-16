import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.op.tensorop import Augmentation2D, Minmax, ModelOp, SparseCategoricalCrossentropy


def MyGen(x, y):
    while True:
        num_data = y.shape[0]
        idx = np.random.randint(0, num_data)
        yield {"x": x[idx], "y": y[idx]}


if __name__ == "__main__":
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)
    data = {"train": lambda: MyGen(x_train, y_train), "eval": lambda: MyGen(x_eval, y_eval)}
    pipeline = fe.Pipeline(
        data=data,
        batch_size=32,
        ops=[
            Augmentation2D(inputs="x",
                           outputs="x",
                           mode="train",
                           rotation_range=10,
                           shear_range=10,
                           zoom_range=0.8,
                           width_shift_range=2,
                           height_shift_range=2),
            Minmax(inputs="x", outputs="x")
        ])
    pipeline.benchmark()
