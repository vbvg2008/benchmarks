import time

import jax.numpy as jnp
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

    batch_size = 64
    num_classes = 1001
    input_shape = (224, 224, 3, batch_size)
    step_size = 0.1
    num_steps = 1000

    init_fun, predict_fun = ResNet50(num_classes)
    import pdb
    pdb.set_trace()
    _, init_params = init_fun(rng_key, input_shape)

    def loss(params, batch):
        inputs, targets = batch
        logits = predict_fun(params, inputs)
        return -jnp.sum(logits * targets)

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = jnp.argmax(targets, axis=-1)
        predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
        return jnp.mean(predicted_class == target_class)

    def synth_batches():
        rng = npr.RandomState(0)
        while True:
            images = rng.rand(*input_shape).astype('float32')
            labels = rng.randint(num_classes, size=(batch_size, 1))
            onehot_labels = labels == jnp.arange(num_classes)
            yield images, onehot_labels

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    batches = synth_batches()

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    opt_state = opt_init(init_params)
    start_time = time.time()
    for i in range(num_steps):
        opt_state = update(i, opt_state, next(batches))
        if i % 100 == 99:
            elapse_time = time.time() - start_time
            print("step: {}, img/sec: {}".format(i, batch_size * 100 / elapse_time))
            start_time = time.time()

    trained_params = get_params(opt_state)
"""
results:
On GTX 1080ti, batch size 64, 209 image/sec


"""
