import pdb

import numpy as np
import tensorflow as tf

data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ds = tf.data.Dataset.from_tensor_slices(data)
ds = ds.shuffle(10)
ds = ds.repeat()

# pdb.set_trace()

# for _ in range(2):
for q in ds:
    print(q)
