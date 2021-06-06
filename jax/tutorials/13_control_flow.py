"""
using jax.grad without jit would work normally just like pytorch

"""
from jax import jit, lax


def f(x):
    print("rebuilding the graph...")
    if x < 3:
        return 3. * x**2
    else:
        return -4 * x


f = jit(f, static_argnums=(0, ))

print(f(2.))  # the graph will be built at compile time, if the value changes, retracing is required.
print(f(4))
print(f(4))
# f2 = jit(f)
# print(f2(2.))  # the graph will be built at compile time, if the value changes, retracing is required.
# print(f2(4))
"""
Re-writing control-flow in other logics
lax.cond -> differentiable
max.while_loop: forward-mode differentiable
lax.fori_loop: forward-mode differentiable
lax.scan: differentiable
"""
@jit
def f2(x):
    print("rebuilding the graph...")
    return lax.cond(x < 3, lambda x: 3 * x**2, lambda x: -4 * x, x)


print(f2(2))
print(f2(4))  # graph not building, yay!
print(f2(4))  # graph not building, yay!
