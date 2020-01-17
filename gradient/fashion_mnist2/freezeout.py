import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential, layers

import fastestimator as fe
from fastestimator.architecture import LeNet
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import Gradients, ModelOp, Scale, SparseCategoricalCrossentropy, UpdateOp
from fastestimator.trace import Accuracy, ModelSaver
from fastestimator.util.util import to_list


def ResNet101(input_shape=(28, 28, 1), classes=10):
    """Creates a LeNet model with 3 convolutional and 2 dense layers.

    Args:
        input_shape (tuple, optional): shape of the input data. Defaults to (28, 28, 1).
        classes (int, optional): number of classes. Defaults to 10.

    Returns:
        'Model' object: LeNet model.
    """
    inputs = layers.Input(shape=input_shape)
    resnet101 = tf.keras.applications.ResNet101(weights=None, include_top=False, input_tensor=inputs, pooling="avg")
    x = resnet101.outputs[0]
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


class Freezeout(TensorOp):
    def __init__(self, layer_idx, prob, model, inputs=None, outputs=None, mode=None):
        """Freeze out the input node of layers, currently only working on dense layer
        Args:
            layer_idx (int, list): layer index or list of layers to freezeout the input of
            prob (float, list): probability of freezing out, must be list if layer_idx is list
            model (Model): keras model instance
            inputs (str, optional): inputs of operator, normally gradient. Defaults to None.
            outputs (str, optional): outputs of operator, normally gradient. Defaults to None.
        """
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.layer_idx = to_list(layer_idx)
        self.prob = to_list(prob)
        if len(self.prob) == 1 and len(self.layer_idx) > 1:
            self.prob = self.prob * len(self.layer_idx)
        self._create_layer_variable_map()
        assert len(self.prob) == len(self.layer_idx), ""

    def _create_layer_variable_map(self):
        layer_var_map = {}
        variable_idx = 0
        for layer_idx, layer in enumerate(self.model.layers):
            var_list_layer = []
            for _ in layer.trainable_variables:
                var_list_layer.append(variable_idx)
                variable_idx += 1
            layer_var_map[layer_idx] = var_list_layer
        self.layer_var_map = layer_var_map

    def forward(self, data, state):
        gradient_var_list = data
        for layer_idx, prob in zip(self.layer_idx, self.prob):
            var_idx_list = self.layer_var_map[layer_idx]
            for var_idx in var_idx_list:
                if len(gradient_var_list[var_idx].shape) == 2:
                    height, width = gradient_var_list[var_idx].shape
                    multiplier = tf.cast(tf.random.uniform([height, 1]) < prob, tf.float32)
                    multiplier = tf.tile(multiplier, [1, width])
                    gradient_var_list[var_idx] = gradient_var_list[var_idx] * multiplier
        return gradient_var_list


def get_estimator(epochs=30, batch_size=192, model_dir=tempfile.mkdtemp()):
    # step 1. prepare data
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.fashion_mnist.load_data()
    train_data = {"x": np.expand_dims(x_train, -1), "y": y_train}
    eval_data = {"x": np.expand_dims(x_eval, -1), "y": y_eval}
    data = {"train": train_data, "eval": eval_data}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Scale(scalar=1.0 / 255, inputs="x", outputs="x"))

    # step 2. prepare model
    model = fe.build(model_def=ResNet101,
                     model_name="resnet101",
                     optimizer="adam",
                     loss_name="loss")

    network = fe.Network(ops=[
        ModelOp(inputs="x", model=model, outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("y", "y_pred"), outputs="loss"),
        Gradients(loss="loss", models=model, outputs="gradients"),
        Freezeout(inputs="gradients", outputs="gradients", layer_idx=347, prob=0.5, model=model, mode="train"),
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
