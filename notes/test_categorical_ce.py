import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.categorical_crossentropy import categorical_crossentropy

# test sparse_categorical_entropy
y_pred = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]], dtype=np.float32)

y_true1 = np.array([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6]], dtype=np.float32)
y_true2 = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.uint8)

y_pred_tf = tf.convert_to_tensor(y_pred)
y_true1_tf = tf.convert_to_tensor(y_true1)
y_true2_tf = tf.convert_to_tensor(y_true2)

y_pred_torch = torch.tensor(y_pred)
y_true1_torch = torch.tensor(y_true1)
y_true2_torch = torch.tensor(y_true2)

print(categorical_crossentropy(y_pred_tf, y_true1_tf, average_loss=False))
print(categorical_crossentropy(y_pred_torch, y_true1_torch, average_loss=False))

print(categorical_crossentropy(y_pred_tf, y_true2_tf, average_loss=False))
print(categorical_crossentropy(y_pred_torch, y_true2_torch, average_loss=False))

print(categorical_crossentropy(y_pred_tf, y_true1_tf, from_logits=True, average_loss=False))
print(categorical_crossentropy(y_pred_torch, y_true1_torch, from_logits=True, average_loss=False))

print(categorical_crossentropy(y_pred_tf, y_true2_tf, from_logits=True, average_loss=False))
print(categorical_crossentropy(y_pred_torch, y_true2_torch, from_logits=True, average_loss=False))
