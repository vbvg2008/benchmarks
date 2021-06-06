import jax.numpy as jnp
from jax import jit

"""
1. jax transformation and compilation only works on functionally pure. A pure function meets:
a. all inputs are passed through function parameters
b. all outputs are passed through function results
c. produce the same result for the same input
"""


def impure_print_side_effect(x):
    print("Executing function")  # This is a side-effect
    return x


# The side-effects appear during the first run
print("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
