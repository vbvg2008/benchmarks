import jax.numpy as jnp
from jax import jit, random, vmap
from util import timeit

key = random.PRNGKey(0)
mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))


def apply_matrix(v):
    return jnp.dot(mat, v)


def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])


@jit
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)


@jit
def vmap_batched_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)


print("naive methods without jit:")
timeit(lambda: naively_batched_apply_matrix(batched_x).block_until_ready())

print("naive methods with jit:")
timeit(lambda: jit(naively_batched_apply_matrix)(batched_x).block_until_ready())

print("jax matrix multiplication:")
timeit(lambda: batched_apply_matrix(batched_x).block_until_ready())

print("vmap auto-vectorization:")
timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready())
"""
naive methods without jit:
After 100 runs, average running time is 4.747381210327148 ms
naive methods with jit:
After 100 runs, average running time is 0.13124942779541016 ms
jax matrix multiplication:
After 100 runs, average running time is 0.08176088333129883 ms
vmap auto-vectorization:
After 100 runs, average running time is 0.05295515060424805 ms
"""
