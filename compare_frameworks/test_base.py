import numpy as np
import tensorflow as tf

# def test_intel_tensorflow():
#     """
#     Check if Intel version of TensorFlow is installed
#     """
#     import tensorflow as tf

#     print("We are using Tensorflow version {}".format(tf.__version__))

#     major_version = int(tf.__version__.split(".")[0])
#     if major_version >= 2:
#         from tensorflow.python import _pywrap_util_port
#         print("Intel-optimizations (DNNL) enabled:", _pywrap_util_port.IsMklEnabled())
#     else:
#         print("Intel-optimizations (DNNL) enabled:", tf.pywrap_tensorflow.IsMklEnabled())

# test_intel_tensorflow()  # Prints if Intel-optimized TensorFlow is used.

in_shape = [1024, 1024, 1]
epochs = 10
batch_size = 16


def my_model_tf(input_shape=in_shape, num_classes=10):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


nTrain = 2000
x_train = np.ones(([nTrain] + in_shape[:2] + [1]))
y_train = np.zeros((nTrain, 10))

nVal = 1000
x_eval = np.ones(([nVal] + in_shape[:2] + [1]))
y_eval = np.zeros((nVal, 10))
