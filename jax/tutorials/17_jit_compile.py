import jax
import jax.numpy as jnp
from util import timeit


def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


@jax.jit
def selu2(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(1000000)

selu3 = jax.jit(selu)
timeit(lambda: selu(x).block_until_ready())
timeit(lambda: jax.jit(selu)(x).block_until_ready())
timeit(lambda: selu2(x).block_until_ready())
timeit(lambda: selu3(x).block_until_ready())

# there is a slight time difference between calling jit(fun)(x) and use jit to decorate the functon

# jit's default level is shaped array, meaning as long as array has the same shape, it will try to re-use the graph
