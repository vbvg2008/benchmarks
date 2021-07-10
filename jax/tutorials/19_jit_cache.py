import jax
import jax.numpy as jnp
from util import timeit


def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


selu = jax.jit(selu, static_argnums=0)

# x = jnp.arange(1000000)

x = 0.1

x1 = x + 1
x2 = x + 2

timeit(lambda: selu(x).block_until_ready(), num_runs=3, warmup=False)
timeit(lambda: selu(x).block_until_ready(), num_runs=3, warmup=False)
timeit(lambda: selu(x1).block_until_ready(), num_runs=3, warmup=False)
timeit(lambda: selu(x2).block_until_ready(), num_runs=3, warmup=False)
timeit(lambda: selu(x).block_until_ready(), num_runs=3, warmup=False)
# timeit(lambda: selu(x).block_until_ready(), num_runs=5, warmup=False)
