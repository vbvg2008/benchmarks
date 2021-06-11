from jax import grad
import jax.numpy as jnp
import jax


def sum_of_squares(x):
  return jnp.sum(x**2)


sum_of_squares_dx = grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))

print(sum_of_squares_dx(x))

#jax.grad will only work on functions with scalar output

"""
# how jax compute grad is fundamentally different from Torch or TF.   Jax works on the function by itself, whereas others
# work on the loss tensor.

But, underlying principle is that jax takes the function and constructs a jaxpr(jax expression), where it's easy to calculate
the gradient analytically.   But tensorflow or Pytorch uses graph to track the operations. and later traverse the graph
"""

# by default, when calling `grad`, the gradient will be calculated w.r.t the first arg.
def sum_squared_error(x, y):
  return jnp.sum((x-y)**2)

sum_squared_error_dx = grad(sum_squared_error)

y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(sum_squared_error_dx(x, y)) # this is calculating gradient w.r.t x

# when trying to calculate derivative for several variables: set argnums

sum_squared_error_dx_and_dy = grad(sum_squared_error, argnums=(0, 1))
print(sum_squared_error_dx_and_dy(x, y))


"""
Don't worry, this doesn't mean everytime we have to write a huge list of arrays as argument.
later we can leverage 'pytrees' to use nested structure for gradient calculation. It looks like:

def loss_fn(params, data):
  ...

grads = jax.grad(loss_fn)(params, data_batch)
"""

# Getting both value and gradient. If you need both val and grad (like when getting loss value),
# then it's more efficient to get them in single function than to call them separately.
print(jax.value_and_grad(sum_squared_error)(x, y)) # for gradient of first arg
print(jax.value_and_grad(sum_squared_error,argnums=(0, 1))(x, y)) # for gradient of both x and y



"""
auxiliary data: sometimes we need more than one output to return, (like auxiliary data), by default, it will give error
when calling grad.  To make the program ignore the second output while calculating gradient, but evaluate both outputs forward,
you can use `has_aux`.
"""

def squared_error_with_aux(x, y):
  return sum_squared_error(x, y), x-y

print(squared_error_with_aux(x,y))
print(grad(squared_error_with_aux, has_aux=True)(x, y)) # it will return grad, then the aux


"""
difference between jax and numpy:
* jax is functional
* Don't write code that has side-effect! side effect is any effect of function that doesn't appear in output, like modifying
array in place.

"""