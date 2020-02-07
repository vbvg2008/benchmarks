import pdb

import tensorflow as tf

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ds = tf.data.Dataset.from_tensor_slices(data)
# ds = ds.repeat()
ds = ds.shuffle(10)

it = iter(ds)
# sample_data = next(it)
# print(sample_data)

# pdb.set_trace()
print("===")
for ele in it:
    print(ele)

print("===")
for ele in it:
    print(ele)
