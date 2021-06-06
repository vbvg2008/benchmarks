# 1. jax execute sequentially by default
# 2. with JIT, sequence of oeprations can be optimized and run together
# 3. not all Jax code can be jit compilied,  compilation requires static shape known at compile time.
import jax.numpy as jnp
import numpy as np
from jax import jit
from util import timeit


def norm(X):
    X = X - X.mean(0)
    return X / X.std(0)


norm_compiled = jit(norm)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)

timeit(lambda: norm(X).block_until_ready())  # 2.77ms
timeit(lambda: norm_compiled(X).block_until_ready())  #0.35ms


def get_negatives(x):
    return x[x < 0]


# this function can not be used as because the shape is not known
x_np = np.random.randn(100)
x_jnp = jnp.array(x_np)

# now let's compare the numpy performance vs jax perfomance on cpu only:
timeit(lambda: get_negatives(x_np), num_runs=1000)  # 0.003 ms
timeit(lambda: get_negatives(x_jnp), num_runs=1000)  # 1.15 ms !

# It seems like when jit is not available, working on numpy array is much faster than working on jnp array.
# maybe this is because of unknown shape?


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x_np = np.random.randn(1000)
x_jnp = jnp.array(x_np)
# now let's compare the numpy performance vs jax perfomance on cpu only:
timeit(lambda: selu(x_np), num_runs=1000)  # 0.9 ms
timeit(lambda: selu(x_jnp), num_runs=1000)  # 1.06 ms !

# when jit is not available, and the shape is static, working on numpy array is still faster than working on jnp array. but not as much.
