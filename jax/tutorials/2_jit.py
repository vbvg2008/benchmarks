import jax.numpy as jnp
from jax import jit, random
from util import timeit


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


@jit
def selu2(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


key = random.PRNGKey(0)
x = random.normal(key, shape=(1000000, ))
timeit(lambda: selu(x).block_until_ready(), num_runs=1, warmup=False)  # first run takes 342 ms
timeit(lambda: selu2(x).block_until_ready(), num_runs=1, warmup=False)  # first run takes 95 ms

timeit(lambda: selu(x).block_until_ready(), num_runs=1000, warmup=True)  # takes 1.59ms
timeit(lambda: selu2(x).block_until_ready(), num_runs=1000, warmup=True)  # takes 0.08ms
