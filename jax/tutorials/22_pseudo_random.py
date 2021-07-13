import jax
import jax.numpy as jnp
import numpy as np

# numpy provides a sequential equivalent guarantee
np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))

#----------------------------------------------
# Jax random requirements:
"""
1. it needs to be reproducible
2. it needs to be parallelizable
3. it needs to be vectorisable
"""
np.random.seed(0)


def bar():
    return np.random.uniform()


def baz():
    return np.random.uniform()


def foo():
    return bar() + 2 * baz()


print(foo())
"""
in order to be reproducible, the bar and baz are executed in specific order, in other words, sequentially. However, it violates the 2nd requirement.

To avoid this, jax does not use a global state, random functions explicitly consume the state, which is referred to as a key.
"""

key = jax.random.PRNGKey(42)

print(key)  # random key is just another word for random seed, but it won't be updated by random function

print(jax.random.normal(key))
print(jax.random.normal(key))

# anoter note: feeding the same key to different random functions can produce correlated outputs, undesirable
# rule of thumb: never reuse keys (unless using identical outputs)

# how to achieve randomness: use key splitting

key, subkey = jax.random.split(key)  # then use sub-key to generate random number, use key to split again.

key, subkey1, subkey2 = jax.random.split(key, num=3)
print(key, subkey1, subkey2)
"""
jax random does not guarantee sequential equivalence:
"""
key = jax.random.PRNGKey(42)
subkeys = jax.random.split(key, 3)
sequence = np.stack([jax.random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = jax.random.PRNGKey(42)
print("all at once: ", jax.random.normal(key, shape=(3, )))

# key should only be used once! (either in random or split), this is called sngle-use principle
