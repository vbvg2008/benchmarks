import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.sparse_categorical_crossentropy import sparse_categorical_crossentropy

# test sparse_categorical_entropy
y_pred = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]], dtype=np.float32)
y_pred1 = np.array([[0.03, 0.07], [0.03, 0.07], [0.03, 0.07], [0.03, 0.07], [0.03, 0.07]], dtype=np.float32)
y_true1 = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
y_true2 = np.array([0, 1, 1, 0, 1], dtype=np.int32)
y_true3 = np.array([0, 1, 1, 0, 1], dtype=np.int64)
y_true4 = np.array([[0], [1], [1], [0], [1]], dtype=np.int32)

y_pred_tf = tf.convert_to_tensor(y_pred)
y_pred1_tf = tf.convert_to_tensor(y_pred1)
y_true1_tf = tf.convert_to_tensor(y_true1)
y_true2_tf = tf.convert_to_tensor(y_true2)
y_true3_tf = tf.convert_to_tensor(y_true3)
y_true4_tf = tf.convert_to_tensor(y_true4)

y_pred_torch = torch.tensor(y_pred)
y_pred1_torch = torch.tensor(y_pred1)
y_true1_torch = torch.tensor(y_true1)
y_true2_torch = torch.tensor(y_true2)
y_true3_torch = torch.tensor(y_true3)
y_true4_torch = torch.tensor(y_true4)

print(sparse_categorical_crossentropy(y_pred_tf, y_true1_tf))
print(sparse_categorical_crossentropy(y_pred_torch, y_true1_torch))

print(sparse_categorical_crossentropy(y_pred_tf, y_true2_tf))
print(sparse_categorical_crossentropy(y_pred_torch, y_true2_torch))

print(sparse_categorical_crossentropy(y_pred_tf, y_true3_tf))
print(sparse_categorical_crossentropy(y_pred_torch, y_true3_torch))

print(sparse_categorical_crossentropy(y_pred_tf, y_true4_tf))
print(sparse_categorical_crossentropy(y_pred_torch, y_true4_torch))

print(sparse_categorical_crossentropy(y_pred1_tf, y_true1_tf, from_logits=True))
print(sparse_categorical_crossentropy(y_pred1_torch, y_true1_torch, from_logits=True))

print(sparse_categorical_crossentropy(y_pred1_tf, y_true2_tf, from_logits=True))
print(sparse_categorical_crossentropy(y_pred1_torch, y_true2_torch, from_logits=True))

print(sparse_categorical_crossentropy(y_pred1_tf, y_true3_tf, from_logits=True))
print(sparse_categorical_crossentropy(y_pred1_torch, y_true3_torch, from_logits=True))

print(sparse_categorical_crossentropy(y_pred1_tf, y_true4_tf, from_logits=True))
print(sparse_categorical_crossentropy(y_pred1_torch, y_true4_torch, from_logits=True))
