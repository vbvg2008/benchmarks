import jax
import jax.numpy as jnp
from jax import jit

global_list = []


@jit
def log2(x):
    global_list.append(
        x)  # to show side-effect not being captured by the expression, therefore won't execute after the graph is built
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2


print(jax.make_jaxpr(log2)(3.0))
# jax first convert the python function into a simple intermediate language called jaxpr

print(log2(4.0))
print(log2(5.0))
print(log2(6.0))
print(global_list)
"""
What happens during tracing:
1. Jax wraps each argument of the function by a tracer oject
2. every time a "JAX" operation is applied on them, tracer object records them
3. Then Jax used tracer records and construct the entire function in jaxpr
4. start to run jaxpr

"""
