import pdb

import jax
import jax.numpy as jnp
from util import timeit

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1:i + 2], w))
    return jnp.array(output)


convolve_jit1 = jax.jit(convolve)

timeit(f=lambda: convolve(x, w))
timeit(f=lambda: convolve_jit1(x, w))

# print(jax.make_jaxpr(convolve_jit1)(x, w))
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])


def mannually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)


def manually_vectorized_convolve(xs, ws):
    output = []
    for i in range(1, xs.shape[-1] - 1):
        output.append(jnp.sum(xs[:, i - 1:i + 2] * ws, axis=1))
    return jnp.stack(output, axis=1)


timeit(lambda: mannually_batched_convolve(xs, ws))
timeit(lambda: manually_vectorized_convolve(xs, ws))

# now start automatica vectorization
auto_batch_convolve = jax.vmap(convolve)
timeit(lambda: auto_batch_convolve(xs, ws))
print(auto_batch_convolve(xs, ws))

# by default, the automatic vectorization works on the first dimension. now we can specify the input&output dimension to vectorize
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)
print(auto_batch_convolve_v2(xst, wst))

# if there is only one argument that requires vectorization, then we can use `in_axes` argument:
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

print(batch_convolve_v3(xs, w))

# jit and vmap can be used together.

jit_vmap_convolve = jax.jit(jax.vmap(convolve))

vmap_jit_convolve = jax.vmap(jax.jit(convolve))
timeit(lambda: jit_vmap_convolve(xs, ws))
timeit(lambda: vmap_jit_convolve(xs, ws))
print(jax.make_jaxpr(jit_vmap_convolve)(xs, ws))

print(jax.make_jaxpr(vmap_jit_convolve)(xs, ws))

# seem like first apply vmap then apply jit is the way to go
