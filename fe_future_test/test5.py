a = {"a": 1, "b": 2, "c": 3, "d": 4}

b = {"a", "b", "e"}


a_keys = set(a.keys())

use_keys = b.intersection(a_keys)

for key in use_keys:
    print(key)
