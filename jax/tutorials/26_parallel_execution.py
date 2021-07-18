"""
this is for single program multiple data (SPMD)

SPMD is basically ensuring the same computation on different input data run in parallel on difference
devices

Basically this is not very different from vectorization (jax.vmap). But among device level, jax.pmap can
help achieve that.
"""
# basic example:
import os
import pdb

import jax
import jax.numpy as jnp
import numpy as np
from util import timeit

# first disable GPU when running on single GPU system, to simulate multi-device behavior on multiple CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

print(jax.devices())

x = np.arange(5)
w = np.array([2., 3., 4.])


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1:i + 2], w))
    return jnp.array(output)


print(convolve(x, w))
# now start parallelism, first prepare some data
n_devices = jax.local_device_count()
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

print(ws)
print(xs)
jit_vmap_convolve = jax.jit(jax.vmap(convolve))
vmap_jit_convolve = jax.vmap(jax.jit(convolve))
# as before, for a vectorized version on single device, we can simply use jax.vmap
timeit(lambda: jit_vmap_convolve(xs, ws))  # still, jit a vmap!   jit a vmap! #0.01ms
timeit(lambda: vmap_jit_convolve(xs, ws))  # 0.4ms

# now start using pmap
pdata = jax.pmap(convolve)(xs, ws)
print(pdata)  # this returns a device array.  During the calculation, each function is called locally

pdata2 = jax.pmap(convolve, in_axes=(0, None))(xs, w)
print(pdata2)

jit_pmap_both = jax.pmap(convolve)  # this is splitting everything
jit_pmap_single = jax.pmap(convolve,
                           in_axes=(0, None))  # this is broadcasting the data to every device, which is less efficient

# only in_axies =0 is supported now
timeit(lambda: jit_pmap_both(xs, ws))  #0.98ms
timeit(lambda: jit_pmap_single(xs, w))  #0.86ms
"""
jax.map automatically compiles the function
"""
