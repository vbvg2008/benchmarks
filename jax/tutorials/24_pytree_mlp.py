import pdb

import jax
import jax.numpy as jnp
import numpy as np
from util import timeit


def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in), bias=np.ones(shape=(n_out, ))))
    return params


def init_mlp_params2(layer_widths):
    key = jax.random.PRNGKey(42)
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        key, sub_key = jax.random.split(key, 2)
        params.append(
            dict(weights=jax.random.normal(sub_key, shape=(n_in, n_out)) * jnp.sqrt(2 / n_in),
                 bias=jnp.ones(shape=(n_out, ))))
    return params


params = init_mlp_params([1, 128, 128, 1])
params2 = init_mlp_params2([1, 128, 128, 1])
print(jax.tree_map(lambda x: x.shape, params))
print(jax.tree_map(lambda x: x.shape, params2))
a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]
print(jax.tree_map(lambda x: jnp.ones(x.shape), a_tree))
"""
Continue with MLP
"""


def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['bias'])
    return x @ last['weights'] + last['bias']


def forward2(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(jax.lax.batch_matmul(x, layer['weights']) + layer['bias'])
    return jax.lax.batch_matmul(x, layer['weights']) + last['bias']


def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y)**2)


@jax.jit
def update(params, x, y, lr=0.1):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_multimap(lambda p, g: p - lr * g, params, grads)


xs = np.random.normal(size=(128, 1))
ys = xs**2

timeit(lambda: update(params2, xs, ys), num_runs=10000)  # this takes 0.15ms per loop
#timeit(lambda: update(params, xs, ys), num_runs=10000)  # this takes 0.31ms per loop
"""
Conclusion:
1. using jnp array as param can be faster, since it does not need to send between cpu and gpu
2. using @ operator is similar to jnp.matmul in terms of speed, surprisingly, jax.lax.batch_matmul is slightly slower
"""
