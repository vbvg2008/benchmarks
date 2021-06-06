from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, make_jaxpr


# jit works similar to @tf.function, it decides its tracing behavior for specific input shape and type
@jit
def f(x, y):
    print("Running f():")
    print(f"  x = {x}")
    print(f"  y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f"  result = {result}")
    return result


# =============with JIT, first time of execution, it builds the graph(jax expression to be accurate), the graph is dependent on input shape and type=======
x = np.random.randn(3, 4)
y = np.random.randn(4)
f(x, y)

# ===============With JIT, second time of execution will execute in XLA instead of in python====================
x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)

# without JIT, the print message executes in python 2 times


def f2(x, y):
    return jnp.dot(x + 1, y + 1)


print(make_jaxpr(f2)(x, y))


@jit
def f3(x, neg):
    return -x if neg else x


# f3(1, True)  # control flow cannot depends on traced values


@partial(jit, static_argnums=(1, ))  # to solve this, we can mark the first argument as non-static
def f4(x, neg):
    print("I am executing")
    return -x if neg else x


print(f4(1, True))  # will compile
print(f4(2, True))  # will not compile, therefore won't print
print(f4(1, False))  # will compile again, since the second argument is not traced
