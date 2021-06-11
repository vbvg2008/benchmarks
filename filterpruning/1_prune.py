import os
import pdb
import tempfile
import time

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_images,
    train_labels,
    epochs=4,
    validation_split=0.1,
)
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1  # 10% of training set will be used for validation set.

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                          final_sparsity=0.80,
                                                          begin_step=0,
                                                          end_step=end_step))
model_for_pruning.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),  # tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
model_for_pruning.fit(train_images,
                      train_labels,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=validation_split,
                      callbacks=callbacks)
_, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Pruned test accuracy:', model_for_pruning_accuracy)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


def benchmark_model(model, num_trials=1000):
    total_time = []
    for i in range(num_trials):
        data = np.random.rand(1, 28, 28, 1)
        start = time.time()
        output = single_inference(model=model, data=data)
        total_time.append(time.time() - start)
        # print("-----{} / {} ----".format(i + 1, num_trials))
    print("Average Inferencing speed is {} ms with {} trials".format(np.mean(total_time[1:]) * 1000, num_trials))


# benchmark_model(model)
# benchmark_model(model_for_export)
# so even though the model size is reduced, the inferencing speed isn't improved much, unless there's a re-formulation of network operation that uses the sparsity.
# like XNNPACK

converter = tf.lite.TFLiteConverter.from_keras_model(model)
dense_tflite_model = converter.convert()
with open("dense.tflite", 'wb') as f:
    f.write(dense_tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
pruned_tflite_model = converter.convert()
with open("sparse.tflite", 'wb') as f:
    f.write(pruned_tflite_model)
