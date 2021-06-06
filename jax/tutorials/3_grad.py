import jax.numpy as jnp
from jax import grad, jit


def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


x_small = jnp.arange(3.0)
grad_fn = grad(grad(sum_logistic))
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
print(grad_fn(x_small))
