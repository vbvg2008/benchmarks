import os
import pdb

import jax
import jax.numpy as jnp
import numpy as np
from util import timeit

# first disable GPU when running on single GPU system, to simulate multi-device behavior on multiple CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
"""
jax.lax.p* like psum, pmax can collectively operate on all devices and return answer
"""
x = np.arange(5)
w = np.array([2., 3., 4.])

# now start parallelism, first prepare some data
n_devices = jax.local_device_count()
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)


def normalized_convolution(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1:i + 2], w))
    output = jnp.array(output)
    return output / jax.lax.psum(output, axis_name="batch")  #this axis name tells it which axisto communicate across


data = jax.pmap(normalized_convolution, axis_name="batch", in_axes=(0, None))(xs, w)
print(data)

# vmap also can use axis_name
data2 = jax.vmap(normalized_convolution, axis_name='batch')(xs, ws)
print(data2)
"""
Important:
pmap and vmap are the only two functions that can specify axis name.

The reason why axis_name is needed is because we might nest pmap and vmap together and we need to make axis clear.

For example:
jax.vmap(jax.pmap(f, axis_name='i'), axis_name='j')

vmap and pmap can be nested in any order, both can also be nested by themselves
"""
