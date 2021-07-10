import jax


def g(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i


g_jit = jax.jit(g)

# Should raise an error.
# g_jit(10, 20)


@jax.jit
def loop_body(prev_i):
    return prev_i + 1


def g_inner_jitted(x, n):
    i = 0
    while i < n:
        i = loop_body(i)
    return x + i


print(g_inner_jitted(10, 20))

# if we need to re-trace a function based on value (instead of shape), then we can use:
g_jit_correct = jax.jit(g, static_argnums=0)  # this is the index of the argument to perform value-trace
print(g_jit_correct(10, 20))
print(g_jit_correct(11, 21))

# if the function isn't going to run multiple times, then it's not worth to jit it
