import time

import jax.numpy as jnp
import numba
import numpy as np
from jax import jit as jax_jit
from numba import jit as numba_jit


def timeit(f, num_runs=100, warmup=True):
    times = []
    if warmup:
        f()  # first call the function for a good luck
    for _ in range(num_runs):
        start = time.time()
        f()
        times.append(time.time() - start)
    print("After {} runs, average running time is {} ms".format(num_runs, 1000 * np.mean(times)))


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)


@jax_jit
def selu_jax(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


@numba_jit(nopython=True)
def selu_numba(x, alpha=1.67, lmbda=1.05):
    return lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)


x = np.random.normal(size=(1000000, ))

timeit(lambda: selu(x), num_runs=1, warmup=False)  # first run takes 37 ms
timeit(lambda: selu_jax(x).block_until_ready(), num_runs=1, warmup=False)  # first run takes 52 ms
timeit(lambda: selu_numba(x), num_runs=1, warmup=False)  # first run takes 609 ms

timeit(lambda: selu(x), num_runs=1000, warmup=True)  # takes 32.85ms
timeit(lambda: selu_jax(x).block_until_ready(), num_runs=1000, warmup=True)  # takes 1.30 ms
timeit(lambda: selu_numba(x), num_runs=1000, warmup=True)  # takes 28.86ms
