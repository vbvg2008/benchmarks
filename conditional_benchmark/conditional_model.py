import pdb
import random
import time

import numpy as np
import tensorflow as tf


@tf.function
def single_inference(model, data):
    output = model(*data, training=False)
    return output


class ConditionalModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = tf.keras.applications.ResNet50(weights=None, include_top=True)

    def call(self, image, condition):
        if condition == 1:
            results = self.backbone(image)
        else:
            results = tf.zeros(shape=(1, 1000), dtype=tf.float32)
        return results


def benchmark(model, p, num_samples=5000):
    conditions = [1] * int(num_samples * p) + [0] * (num_samples - int(num_samples * p))
    random.shuffle(conditions)
    images = [np.random.rand(1, 224, 224, 3) for _ in range(num_samples)]
    times = []
    # warmup
    single_inference(model, (images[0], conditions[0]))
    for image, condition in zip(images, conditions):
        start = time.time()
        output = single_inference(model, (image, condition))
        times.append(time.time() - start)
    print("After {} runs, average running time is {} ms".format(num_samples, 1000 * np.mean(times)))


def benchmark_normal(model, num_samples=5000):
    images = [np.random.rand(1, 224, 224, 3) for _ in range(num_samples)]
    times = []
    # warmup
    single_inference(model, (images[0], ))
    for image in images:
        start = time.time()
        output = single_inference(model, (image, ))
        times.append(time.time() - start)
    print("After {} runs, average running time is {} ms".format(num_samples, 1000 * np.mean(times)))


if __name__ == "__main__":
    model = ConditionalModel()
    benchmark(model, p=0.1)  # 1.928 ms on A100, 2.81 ms on M60
    benchmark(model, p=0.3)  # 2.697 ms on A100, 4.71 ms on M60
    benchmark(model, p=0.5)  # 4.034 ms on A100, 7.29 ms on M60
    benchmark(model, p=0.7)  # 5.895 ms on A100, 11.97 ms on M60
    benchmark(model, p=0.9)  # 7.059 ms on A100, 13.88 ms on M60

    # benchmark normal condition
    model2 = tf.keras.applications.ResNet50(weights=None, include_top=True)
    benchmark_normal(model2)  # 6.70 ms on A100
