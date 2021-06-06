import jax.numpy as jnp

x = jnp.arange(10)
print(id(x))
# x[0] = 10
x = x.at[0].set(10)
print(id(x))
"""jnp can be used for numpy for general purpose. but jnp is immutable.
having said that, jax has index update syntax system to take care of index assignment."""
