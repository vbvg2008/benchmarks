# undertanding and choosing when to use static and when to trace is the key to program efficiently.

import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def f(x):
    return x.reshape(jnp.array(x.shape).prod())


def f2(x):
    return x.reshape(np.array(x.shape).prod())


#reshape requires a static input rather than shapeed value in jit

x = jnp.ones((2, 3))
# f(x)
print(f2(x))

# use numpy for operations that should be static (done at compile time),
# use jnumpy for operations that should be traced (compie and run execute in run-time)

# therefore, a good practice is to use both numpy and jnumpy to have finer control.
