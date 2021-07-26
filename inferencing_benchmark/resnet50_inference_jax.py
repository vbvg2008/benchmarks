import pdb
import time

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import grad, jit, random
from jax.experimental import optimizers, stax
from jax.experimental.stax import AvgPool, BatchNorm, Conv, Dense, FanInSum, FanOut, Flatten, GeneralConv, Identity, \
    LogSoftmax, MaxPool, Relu

# ResNet blocks compose other layers


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(Conv(filters1, (1, 1), strides),
                       BatchNorm(),
                       Relu,
                       Conv(filters2, (ks, ks), padding='SAME'),
                       BatchNorm(),
                       Relu,
                       Conv(filters3, (1, 1)),
                       BatchNorm())
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters

    def make_main(input_shape):
        # the number of output channels depends on the number of input channels
        return stax.serial(Conv(filters1, (1, 1)),
                           BatchNorm(),
                           Relu,
                           Conv(filters2, (ks, ks), padding='SAME'),
                           BatchNorm(),
                           Relu,
                           Conv(input_shape[3], (1, 1)),
                           BatchNorm())

    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks


def ResNet50(num_classes):
    return stax.serial(GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
                       BatchNorm(),
                       Relu,
                       MaxPool((3, 3), strides=(2, 2)),
                       ConvBlock(3, [64, 64, 256], strides=(1, 1)),
                       IdentityBlock(3, [64, 64]),
                       IdentityBlock(3, [64, 64]),
                       ConvBlock(3, [128, 128, 512]),
                       IdentityBlock(3, [128, 128]),
                       IdentityBlock(3, [128, 128]),
                       IdentityBlock(3, [128, 128]),
                       ConvBlock(3, [256, 256, 1024]),
                       IdentityBlock(3, [256, 256]),
                       IdentityBlock(3, [256, 256]),
                       IdentityBlock(3, [256, 256]),
                       IdentityBlock(3, [256, 256]),
                       IdentityBlock(3, [256, 256]),
                       ConvBlock(3, [512, 512, 2048]),
                       IdentityBlock(3, [512, 512]),
                       IdentityBlock(3, [512, 512]),
                       AvgPool((7, 7)),
                       Flatten,
                       Dense(num_classes),
                       LogSoftmax)


if __name__ == "__main__":
    rng_key = random.PRNGKey(0)
    input_shape = (224, 224, 3, 1)
    num_classes = 1001

    init_fun, predict_fun = ResNet50(num_classes)
    _, init_params = init_fun(rng_key, input_shape)

    @jit
    def single_inference(params, inputs):
        logits = predict_fun(params, inputs)
        return logits

    num_trials = 100
    total_time = []
    for i in range(num_trials):
        data = np.random.rand(224, 224, 3, 1)
        start = time.time()
        result = single_inference(init_params, data)
        total_time.append(time.time() - start)
        # print("-----{} / {} ----".format(i + 1, num_trials))
    print("Average Inferencing speed is {} ms with {} trials".format(np.mean(total_time[1:]) * 1000, num_trials))
    # GPU: 5ms, CPU: 45 ms
