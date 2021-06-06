import pdb
import time

import jax.numpy as jnp
import numpy as np
from jax import device_put, grad, jit, random, vmap
from jax._src.dtypes import dtype

key = random.PRNGKey(0)
# x = random.normal(key, (10, ))
# pdb.set_trace()
# print(x)

size = 3000

# 1==============================================================================
x = random.normal(key, (size, size), dtype=jnp.float32)
jnp.dot(x, x.T).block_until_ready()  # run for one time
times_1 = []
for _ in range(100):
    start = time.time()
    jnp.dot(x, x.T).block_until_ready()  # time it
    times_1.append(time.time() - start)
print("took {} seconds in average".format(np.mean(times_1)))  # 21ms on average on 1080ti

# 2==============================================================================
x2 = np.random.normal(size=(size, size)).astype("float32")
jnp.dot(x2, x2.T).block_until_ready()  # run for one time
times_2 = []
for _ in range(100):
    start = time.time()
    jnp.dot(x2, x2.T).block_until_ready()  # time it
    times_2.append(time.time() - start)
print("took {} seconds in average".format(np.mean(times_2)))  # 33ms on average on 1080ti

# 3==============================================================================
x3 = device_put(np.random.normal(size=(size, size)).astype("float32"))
jnp.dot(x3, x3.T).block_until_ready()  # run for one time
times_3 = []
for _ in range(100):
    start = time.time()
    jnp.dot(x3, x3.T).block_until_ready()  # time it
    times_3.append(time.time() - start)
print("took {} seconds in average".format(np.mean(times_3)))  # 6ms on average on 1080ti !
