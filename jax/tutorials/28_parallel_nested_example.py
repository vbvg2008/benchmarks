import functools
import os
import pdb
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from util import timeit

# first disable GPU when running on single GPU system, to simulate multi-device behavior on multiple CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray


def init(rng_key):
    """Returns the initial model params"""
    weights_key, bias_key = jax.random.split(rng_key)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)


def loss_fn(params, xs, ys):
    pred = params.weight * xs + params.bias
    return jnp.mean((pred - ys)**2)


#the argument of the function is in [num_devices, batch_per_device, ..]
@functools.partial(jax.pmap, axis_name='device_axis')
def update(params, xs, ys, lr=0.005):
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

    # loss and gradients are now per-device
    grads = jax.lax.pmean(grads, axis_name="device_axis")
    loss = jax.lax.pmean(loss, axis_name="device_axis")

    # perform the update with pytree
    new_params = jax.tree_multimap(lambda param, g: param - g * lr, params, grads)
    return new_params, loss


"""
pmap essentially creates a new dimension of the data
"""
true_w, true_b = 2, -1
xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * true_w + true_b + noise
params = init(jax.random.PRNGKey(123))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)

# at this time, the parameter is still not distributed yet. but when update is first called, each copy will sty on
# its device.    DeviceArray -> not yet distributed,  SharedDeviceArray -> Distributed


# this splits the array from [batch, ...] into [n_device, batch / n_device, ...]
def split(arr):
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


x_split = split(xs)
y_split = split(ys)


def type_after_update(name, obj):
    print(f"after first `update()`, `{name}` is a", type(obj))


for i in range(1000):
    replicated_params, loss = update(replicated_params, x_split, y_split)
    if i == 0:
        type_after_update('replicated_params.weight', replicated_params.weight)
        type_after_update('loss', loss)
        type_after_update('x_split', x_split)
    if i % 100 == 0:
        print("step: {}, loss: {}".format(i, loss[0]))  # each device will return its own loss
