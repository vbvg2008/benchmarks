import jax
import jax.numpy as jnp
from util import timeit

# higher order diff on univariate function
f = lambda x: x**3 + 2 * x**2 - 3 * x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)
print(dfdx(0.))
print(d2fdx(0.))
print(d3fdx(0.))
print(d4fdx(0.))


# multivariate higher order diff
def hessian_forward(f):
    return jax.jacfwd(jax.grad(f))


def hessian_reverse(f):
    return jax.jacrev(jax.grad(f))


f2 = lambda x: jnp.dot(x, x)

f3 = jax.jit(hessian_forward(f2))
f4 = jax.jit(hessian_reverse(f2))
timeit(lambda: f3(jnp.array([x - 0.1 for x in range(100)])))
timeit(lambda: f4(jnp.array([x - 0.1 for x in range(100)])))

#higher order optimization -> allows us to get gradient of update w.r.t parameter
"""
def meta_loss_fn(params, data):
    grads = jax.grad(loss_fn)(params, data)
    return loss_fn(params - lr * grads, data)

meta_grads = jax.grad(meta_loss_fn)(params, data)
"""
# use jax.lax.stop_gradient to stop gradient

# also when it's needed to straight-through the gradient ( connect the gradient through non-differentiable block), one can use jax.lax.stop_gradient
