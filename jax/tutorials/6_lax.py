# jax.numpy as higher level API,  jax.lax is lower level API with stricter rules in usage, but often faster because of it

import jax.numpy as jnp
from jax import lax

jnp.add(1, 1.0)

lax.add(1, 2)

# lax API also provides efficient APIs

x = jnp.array([1, 2, 1])
y = jnp.ones(10)
print(jnp.convolve(x, y))
print(x)
print(y)

# this is a batched convolution operation
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float), y.reshape(1, 1, 10), window_strides=(1, ), padding=[(len(y) - 1, len(y) - 1)])
print(result)

# in general, jax.lax is python wrapper for xla operations.  lax.conv_general_dilated is designed for deep learning convolution.
# jax.lax is the reason jit complication can be used.


