import numpy as np
from jax import random
"""
the default random behavior of numpy changes its random seed behind the hood, and it creates problems with multiprocessing
, this is why jax random numbers are handled explicitly.
"""

key = random.PRNGKey(0)
print(key)  # [0 0] random state is described by two uint32 numbers, call key

# the jax random functions produce pseudorandom number from state, but in contrast, do not change state.

print(random.normal(key, shape=(1, )))
print(key)
# No no no!
print(random.normal(key, shape=(1, 10)))
print(random.normal(key, shape=(1, )))
print(key)

# therefore, the key needs to be changed explicitly everytime it's used.
# the process is called split key
print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1, ))
print("    \---SPLIT --> new key   ", key)
print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)

print("old key", key)
key2 = random.split(key, num=1)
normal_pseudorandom = random.normal(key, shape=(1, ))
print("    old key   ", key)
print("  new subkey", key2, "--> normal", normal_pseudorandom)
