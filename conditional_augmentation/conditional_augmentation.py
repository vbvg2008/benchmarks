import numpy as np
import tensorflow as tf

import fastestimator as fe
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Augmentation2D, Minmax


class ConditionalAugmentation(TensorOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.aug2d = Augmentation2D(inputs=inputs,
                                    outputs=outputs,
                                    mode=mode,
                                    rotation_range=10,
                                    shear_range=10,
                                    zoom_range=0.8,
                                    width_shift_range=2,
                                    height_shift_range=2)

    def forward(self, data, state):
        random_number = tf.random.uniform([])
        if random_number > 0.5:
            data = self.aug2d.forward(data, state)
        return data


if __name__ == "__main__":
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)
    data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
    pipeline = fe.Pipeline(
        data=data,
        batch_size=32,
        ops=[ConditionalAugmentation(inputs="x", outputs="x", mode="train"), Minmax(inputs="x", outputs="x")])
    pipeline.benchmark()
