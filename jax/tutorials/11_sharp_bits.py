# element assignment

# Out of bounds indexing, the retrieval of out-of-bounds array will use the value at boundary instead
# other out-of-bounds operations will be skipped

# passing list or tuple to traced functions can lead to silent performance damage. Therefore, Jax prevents the implicit
# conversion of lists to arrays
import jax.numpy as jnp
import numpy as np
from jax import jit, make_jaxpr
from util import timeit


@jit
def permissive_sum(x):
    return jnp.sum(jnp.array(x))


x = list(range(10))
x2 = np.array(list(range(10)))
# print(permissive_sum(x))
"""
the above permissive_sum has a performance issue when passing a list, each element is treated as separable variables
and processed separately. This can be diagnosed through `make_jaxpr`
"""
print("Without converting to numpy array")
print(make_jaxpr(permissive_sum)(x))
timeit(lambda: permissive_sum(x))  # 0.31ms
print("==============================================")
print("converting to numpy array")
print(make_jaxpr(permissive_sum)(x2))
timeit(lambda: permissive_sum(x2))  # 0.1 ms , which is more efficient.
