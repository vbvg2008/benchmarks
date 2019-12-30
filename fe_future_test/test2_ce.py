import pdb

import numpy as np
import tensorflow as tf
import torch


def compare_ce(batch_size, num_classes, softmaxed=True):
    y_pred_source = np.random.rand(batch_size, num_classes)
    if softmaxed:
        batch_sum = np.reshape(np.sum(y_pred_source, axis=-1), (-1, 1))
        y_pred_source = y_pred_source / np.tile(batch_sum, (1, num_classes))
    y_gt_source = np.random.randint(0, num_classes, size=(batch_size))

    y_pred_tf = tf.convert_to_tensor(y_pred_source)
    y_gt_tf = tf.convert_to_tensor(y_gt_source)
    ce_tf_1a = tf.losses.sparse_categorical_crossentropy(y_gt_tf, y_pred_tf, from_logits=not softmaxed)
    print("ce_tf: {}".format(ce_tf_1a))

    y_pred_torch = torch.tensor(y_pred_source)
    y_gt_torch = torch.tensor(y_gt_source, dtype=torch.long)
    if not softmaxed:
        ce_torch_1a = torch.nn.CrossEntropyLoss(reduction="none")(y_pred_torch, y_gt_torch)
    else:
        ce_torch_1a = torch.nn.NLLLoss(reduction="none")(torch.log(y_pred_torch), y_gt_torch)
    print("ce_torch: {}".format(ce_torch_1a))
    # pdb.set_trace()


#case 1: binary cross-entropy
compare_ce(batch_size=4, num_classes=2, softmaxed=True)
compare_ce(batch_size=4, num_classes=2, softmaxed=False)
