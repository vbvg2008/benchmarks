import tensorflow as tf


@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

print(f(tf.random.uniform([5])))