import numpy as np
import tensorflow as tf
import torch

from fastestimator.backend.binary_crossentropy import binary_crossentropy

# test binary cross_entropy

#segmentation
y_pred1 = np.array([[0.3, 0.7, 0.2], [0.3, 0.7, 0.2], [0.3, 0.7, 0.2]], dtype=np.float32)
y_true1 = np.array([[0.5, 0.7, 0.5], [0.5, 0.7, 0.5], [0.5, 0.7, 0.3]], dtype=np.float32)
y_true2 = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int32)

y_pred1_tf = tf.convert_to_tensor(y_pred1)
y_true1_tf = tf.convert_to_tensor(y_true1)
y_true2_tf = tf.convert_to_tensor(y_true2)
y_pred1_torch = torch.tensor(y_pred1)
y_true1_torch = torch.tensor(y_true1)
y_true2_torch = torch.tensor(y_true2)

#---------
binary_crossentropy(y_pred1_tf, y_true1_tf, average_loss=False)
binary_crossentropy(y_pred1_torch, y_true1_torch, average_loss=False)
binary_crossentropy(y_pred1_tf, y_true1_tf, average_loss=True)
binary_crossentropy(y_pred1_torch, y_true1_torch, average_loss=True)

binary_crossentropy(y_pred1_tf, y_true1_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred1_torch, y_true1_torch, average_loss=False, from_logits=True)
binary_crossentropy(y_pred1_tf, y_true1_tf, average_loss=True, from_logits=True)
binary_crossentropy(y_pred1_torch, y_true1_torch, average_loss=True, from_logits=True)

#----------
binary_crossentropy(y_pred1_tf, y_true2_tf, average_loss=False)
binary_crossentropy(y_pred1_torch, y_true2_torch, average_loss=False)
binary_crossentropy(y_pred1_tf, y_true2_tf, average_loss=True)
binary_crossentropy(y_pred1_torch, y_true2_torch, average_loss=True)

binary_crossentropy(y_pred1_tf, y_true2_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred1_torch, y_true2_torch, average_loss=False, from_logits=True)
binary_crossentropy(y_pred1_tf, y_true2_tf, average_loss=True, from_logits=True)
binary_crossentropy(y_pred1_torch, y_true2_torch, average_loss=True, from_logits=True)

# binary classification
y_pred2 = np.array([[0.3], [0.7], [0.5]], dtype=np.float32)
y_true3 = np.array([[1], [0], [1]], dtype=np.uint8)
y_true4 = np.array([[1.0], [0.0], [1.0]], dtype=np.float64)
y_true5 = np.array([1, 0, 1], dtype=np.uint8)
y_true6 = np.array([1.0, 0.0, 1.0], dtype=np.float64)

y_pred2_tf = tf.convert_to_tensor(y_pred2)
y_true3_tf = tf.convert_to_tensor(y_true3)
y_true4_tf = tf.convert_to_tensor(y_true4)
y_true5_tf = tf.convert_to_tensor(y_true5)
y_true6_tf = tf.convert_to_tensor(y_true6)

y_pred2_torch = torch.tensor(y_pred2)
y_true3_torch = torch.tensor(y_true3)
y_true4_torch = torch.tensor(y_true4)
y_true5_torch = torch.tensor(y_true5)
y_true6_torch = torch.tensor(y_true6)

binary_crossentropy(y_pred2_tf, y_true3_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred2_torch, y_true3_torch, average_loss=False, from_logits=True)
binary_crossentropy(y_pred2_tf, y_true4_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred2_torch, y_true4_torch, average_loss=False, from_logits=True)

binary_crossentropy(y_pred2_tf, y_true5_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred2_torch, y_true5_torch, average_loss=False, from_logits=True)

binary_crossentropy(y_pred2_tf, y_true6_tf, average_loss=False, from_logits=True)
binary_crossentropy(y_pred2_torch, y_true6_torch, average_loss=False, from_logits=True)
