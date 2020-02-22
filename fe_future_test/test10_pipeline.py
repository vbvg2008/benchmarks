import pdb

import tensorflow as tf

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ds = tf.data.Dataset.from_tensor_slices(data)
# ds = ds.repeat()
ds = ds.shuffle(10)

# sample_data = next(it)
# print(sample_data)

it = iter(ds)
print("===")
while True:
    try:
        print(next(it))
    except StopIteration:
        break

print("===")
for ele in ds:
    print(ele)
