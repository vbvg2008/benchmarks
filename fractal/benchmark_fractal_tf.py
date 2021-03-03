import pdb
import sys
import time

import numpy as np
import tensorflow as tf
from fractal_tf import mymodel


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


def benchmark_model(num_blocks, block_level, model, input_shape, num_trials=1000):
    total_time = []
    data = np.random.rand(*input_shape)
    data = np.expand_dims(data, axis=0)
    for i in range(num_trials):
        start = time.time()
        _ = single_inference(model=model, data=data)
        total_time.append(time.time() - start)
        # if i % 100 == 0:
        #     print("-----{} / {} ----".format(i + 1, num_trials))
    average_time = np.mean(total_time[1:]) * 1000
    print("Average Inferencing time for {} blocks and block level {} is {} ms with {} trials".format(
        num_blocks, block_level, average_time, num_trials))


def param_count(model):
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return trainable_count


# def param_estimate(input_shape, num_blocks, num_classes, init_filter=64):
#     h, w, c = input_shape
#     num_param = 0
#     for block_idx in range(num_blocks):
#         if block_idx == 0:
#             input_channel = c
#         else:
#             input_channel = init_filter * 2**(block_idx - 1)
#         num_param += 1 * 1 * input_channel * init_filter * 2**block_idx
#         num_param += 3 * 3 * init_filter * 2**block_idx * init_filter * 2**block_idx
#         num_param += 1 * 1 * init_filter * 2**block_idx * init_filter * 2**block_idx

#     num_param += h * w * init_filter * 2**block_idx * num_classes
#     return num_param

if __name__ == "__main__":
    args = sys.argv[1:]
    num_blocks = int(args[0])
    block_level = int(args[1])
    input_shape = (512, 512, 3)
    num_classes = 1000
    # num_param = param_estimate(input_shape, num_blocks, num_classes)
    # print("number of block: {}, parameter estimate: {}".format(num_blocks, num_param))
    fractal_model = mymodel(num_blocks=num_blocks,
                            block_level=block_level,
                            input_shape=input_shape,
                            num_classes=num_classes)
    # num_param = param_count(fractal_model)
    # print("{} blocks and block level {} has {} parameters".format(num_blocks, block_level, num_param))
    benchmark_model(num_blocks, block_level, model=fractal_model, input_shape=input_shape)
