import tensorflow as tf
import torch
import numpy as np
import pdb


def compare_ce(batch_size, num_classes):
    y_pred_source = np.random.rand(batch_size, num_classes)
    y_gt_source = np.random.randint(0, num_classes, size=(batch_size))

    y_pred_tf = tf.convert_to_tensor(y_pred_source)
    y_gt_tf = tf.convert_to_tensor(y_gt_source)
    ce_tf_1a = tf.losses.sparse_categorical_crossentropy(y_gt_tf, y_pred_tf)
    print("ce_tf: {}".format(ce_tf_1a))

    y_pred_torch = torch.tensor(y_pred_source)
    y_gt_torch = torch.tensor(y_gt_source, dtype=torch.long)
    # pdb.set_trace()
    ce_torch_1a = torch.nn.CrossEntropyLoss(reduction="none")(y_pred_torch, y_gt_torch)
    print("ce_torch: {}".format(ce_torch_1a))

#case 1: binary cross-entropy
compare_ce(batch_size=10, num_classes=10)