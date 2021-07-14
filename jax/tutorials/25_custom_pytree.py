import pdb

import jax


class MyContainer:
    """A named container."""
    def __init__(self, name: str, a: int, b: int, c: int):
        self.name = name
        self.a = a
        self.b = b
        self.c = c


print(jax.tree_leaves([MyContainer('Alice', 1, 2, 3), MyContainer('Bob', 4, 5, 6)]))

# this will give error, because pytree doesn't know how to flatten a custom object
#jax.tree_map(lambda x: x + 1, [MyContainer('Alice', 1, 2, 3), MyContainer('Bob', 4, 5, 6)])

# in order to flatten this again, you can register the container with jax and tell it how to flatten and unflatten it:


def flatten_MyContainer(container):
    flaten_contents = [container.a, container.b, container.c]
    return flaten_contents, container.name  #this function should return a pair


def unflatten_MyContainer(flat_contents, name):
    return MyContainer(name, *flat_contents)  #this function should take a pair


jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)

print(jax.tree_leaves([MyContainer('Alice', 1, 2, 3), MyContainer('Bob', 4, 5, 6)]))

# jax.tree treats None as node without children, not as a leaf

# convert a list of tree into a tree of list:, we can use jax.tree_multimap


def tree_transpose(list_of_trees):
    return jax.tree_multimap(lambda *xs: list(xs), *list_of_trees)


episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
print(tree_transpose(episode_steps))
