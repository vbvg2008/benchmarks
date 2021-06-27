import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jax import numpy as jnp

xs = np.random.normal(size=(100, ))
noise = np.random.normal(scale=0.1, size=(100, ))
ys = xs * 3 - 1 + noise


def model(theta, x):
    w, b = theta
    return w * x + b


def loss_fn(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction - y)**2)


@jit
def update(theta, x, y, lr=0.1):
    return theta - lr * grad(loss_fn)(theta, x, y)


theta = jnp.array([1., 1.])

for _ in range(1000):
    theta = update(theta, xs, ys)

# plt.scatter(xs, ys)
# plt.plot(xs, model(theta, xs))

w, b = theta
print(f"w: {w:<.2f}, b: {b:<.2f}")
