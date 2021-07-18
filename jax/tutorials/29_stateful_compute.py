"""
Use functional programming to handle states in jax program.

example of state in program includes: model parameters, optimizer state, and stateful layers such as BatchNorm
"""

# Simple example, Counter
import jax
import jax.numpy as jnp


class Counter:
    """A simple counter."""
    def __init__(self):
        self.n = 0

    def count(self) -> int:
        """Increments the counter and returns the new value."""
        self.n += 1
        return self.n

    def reset(self):
        """Resets the counter to zero."""
        self.n = 0


counter = Counter()

for _ in range(3):
    print(counter.count())

counter.reset()
fast_count = jax.jit(counter.count)

for _ in range(3):
    print(fast_count())  # ooops
"""
Solution1: Explicit State
so make the state as the input arg of hte method
"""


class CounterV2:
    def count(self, n):
        # You could just return n+1, but here we separate its role as
        # the output and as the counter state for didactic purposes.
        return n + 1, n + 1

    def reset(self):
        return 0


counter = CounterV2()
state = counter.reset()

for _ in range(3):
    value, state = counter.count(state)
    print(value)
state = counter.reset()
fast_count = jax.jit(counter.count)

for _ in range(3):
    value, state = fast_count(state)
    print(value)
"""
general strategy:

make every stateful function into stateless function by class handling
"""


class StatelessClass:
    def stateless_method(state, *args, **kwargs):
        pass


# stateless might always be the nature of jax. To overcome this one might need to explore good libraries that can do
# so easily.
