import pdb
import time

import numpy as np
import tensorflow as tf

from skimage.feature import local_binary_pattern


def Minmax_tensor(data):
    x = tf.cast(data["x"], tf.float32)
    data["x"] = tf.math.divide(tf.subtract(x, tf.reduce_min(x)),
                               tf.maximum(tf.subtract(tf.reduce_max(x), tf.reduce_min(x)), 1e-7))
    return data


def Minmax_numpy(data):
    x = data["x"]
    x_max = np.max(x)
    x_min = np.min(x)
    data["x"] = (x - x_min) / max((x_max - x_min), 1e-7)
    return data


def Minmax_ds(data):
    x = tf.cast(data["x"], tf.float32)
    data["x"] = tf.py_function(func=Minmax, inp=[x], Tout=tf.float32)
    return data


def Minmax(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / max((x_max - x_min), 1e-7)
    return x


def lbp(data):
    x = data["x"].numpy()
    for idx in range(32):
        x[idx] = local_binary_pattern(x[idx], 24, 3)
    data["x"] = x
    return data


def lbp_ds(data):
    x = data["x"]
    data["x"] = tf.py_function(func=local_binary_pattern, inp=[x, 24, 3], Tout=tf.float32)
    return data


(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()

data = {"x": x_train, "y": y_train}

ds = tf.data.Dataset.from_tensor_slices(data)
ds = ds.shuffle(60000)
ds = ds.repeat()
# ds = ds.map(Minmax_tensor, num_parallel_calls=4)
ds = ds.map(lbp_ds, num_parallel_calls=4)
ds = ds.batch(32)
ds = ds.prefetch(1)

tic = time.perf_counter()
for idx, elem in enumerate(ds.take(10000)):
    # elem = lbp(elem)
    if idx > 0 and idx % 100 == 0:
        elapse = time.perf_counter() - tic
        example_sec = 100 * 32 / elapse
        print("step: {}, image/sec: {}".format(idx, example_sec))
        tic = time.perf_counter()
