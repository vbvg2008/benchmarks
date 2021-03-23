import pdb
import sys
import time

import numpy as np
import tensorflow as tf


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


def benchmark_model(level, model, input_shape, num_trials=500):
    total_time = []
    data = np.random.rand(*input_shape)
    data = np.expand_dims(data, axis=0)
    for i in range(num_trials):
        start = time.time()
        _ = single_inference(model=model, data=data)
        total_time.append(time.time() - start)
        if i % 100 == 0:
            print("-----{} / {} ----".format(i + 1, num_trials))
    average_time = np.mean(total_time[1:]) * 1000
    print("Average Inferencing time for level {} is {} ms with {} trials".format(level, average_time, num_trials))


def param_count(model):
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return trainable_count


if __name__ == "__main__":
    args = sys.argv[1:]
    level = int(args[0])
    model_fn = {
        0: tf.keras.applications.EfficientNetB0,
        1: tf.keras.applications.EfficientNetB1,
        2: tf.keras.applications.EfficientNetB2,
        3: tf.keras.applications.EfficientNetB3,
        4: tf.keras.applications.EfficientNetB4,
        5: tf.keras.applications.EfficientNetB5,
        6: tf.keras.applications.EfficientNetB6,
        7: tf.keras.applications.EfficientNetB7
    }
    model = model_fn[level](weights=None)
    num_param = param_count(model)
    input_shape = model.input_shape[1:]
    print("level: {}, num_param: {}, input_shape: {}".format(level, num_param, input_shape))
    benchmark_model(level, model=model, input_shape=input_shape)
