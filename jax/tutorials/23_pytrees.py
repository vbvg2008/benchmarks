"""
pytree is like a nested strucures. pytree is a container of leaf elements and/or more pytrees

Containers include lists, tuples and dicts
"""
import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {
        'k1': 2, 'k2': (3, 4)
    }, 5],
    {
        'a': 2, 'b': (2, 3)
    },
    jnp.array([1, 2, 3]), ]

# Let's see how many leaves they have:
for pytree in example_trees:
    leaves = jax.tree_leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

# in deep learning, we need pytree to express things like model parameters
# common pytree functions: jax.tree_map and jax.tree_multimap
list_of_lists = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
print(jax.tree_map(lambda x: x * 2, list_of_lists))

another_list_of_list = list_of_lists
print(jax.tree_multimap(lambda x, y: x + y, list_of_lists,
                        another_list_of_list))  # when using tree_multimap, the structure must exactly match.
