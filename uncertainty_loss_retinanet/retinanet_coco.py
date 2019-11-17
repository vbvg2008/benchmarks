# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tempfile
from ast import literal_eval

import tensorflow as tf

import fastestimator as fe
from fastestimator.architecture.retinanet import PredictBox, RetinaNet, get_fpn_anchor_box, get_target
from fastestimator.architecture.uncertaintyloss import UncertaintyLoss
from fastestimator.dataset.mscoco import load_data
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ImageReader, ResizeImageAndBbox, TypeConverter
from fastestimator.op.tensorop import Loss, ModelOp, Pad, Rescale
from fastestimator.trace import MeanAvgPrecision, ModelSaver


class String2List(NumpyOp):
    # this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data, state):
        data = map(literal_eval, data)
        return data


class GenerateTarget(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.anchorbox, _ = get_fpn_anchor_box(input_shape=(512, 512, 3))

    def forward(self, data, state):
        obj_label, x1, y1, width, height = data
        cls_gt, x1_gt, y1_gt, w_gt, h_gt = get_target(self.anchorbox, obj_label, x1, y1, width, height)
        return cls_gt, x1_gt, y1_gt, w_gt, h_gt


class RetinaLoss(Loss):
    def focal_loss(self, cls_gt_example, cls_pred_example, alpha=0.25, gamma=2.0):
        # cls_gt_example shape: [A], cls_pred_example shape: [A, K]
        num_classes = cls_pred_example.shape[-1]
        # gather the objects and background, discard the rest
        anchor_obj_idx = tf.where(tf.greater_equal(cls_gt_example, 0))
        anchor_obj_bg_idx = tf.where(tf.greater_equal(cls_gt_example, -1))
        anchor_obj_count = tf.cast(tf.shape(anchor_obj_idx)[0], tf.float32)
        cls_gt_example = tf.one_hot(cls_gt_example, num_classes)
        cls_gt_example = tf.gather_nd(cls_gt_example, anchor_obj_bg_idx)
        cls_pred_example = tf.gather_nd(cls_pred_example, anchor_obj_bg_idx)
        cls_gt_example = tf.reshape(cls_gt_example, (-1, 1))
        cls_pred_example = tf.reshape(cls_pred_example, (-1, 1))
        # compute the focal weight on each selected anchor box
        alpha_factor = tf.ones_like(cls_gt_example) * alpha
        alpha_factor = tf.where(tf.equal(cls_gt_example, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(cls_gt_example, 1), 1 - cls_pred_example, cls_pred_example)
        focal_weight = alpha_factor * focal_weight**gamma / anchor_obj_count
        cls_loss = tf.losses.BinaryCrossentropy(reduction='sum')(cls_gt_example,
                                                                 cls_pred_example,
                                                                 sample_weight=focal_weight)
        return cls_loss, anchor_obj_idx

    def smooth_l1(self, loc_gt_example, loc_pred_example, anchor_obj_idx, beta=0.1):
        # loc_gt is padded x 4, loc_pred is #num_anchor x 4
        loc_pred = tf.gather_nd(loc_pred_example, anchor_obj_idx)  #anchor_obj_count x 4
        anchor_obj_count = tf.shape(loc_pred)[0]
        loc_gt = loc_gt_example[:anchor_obj_count]  #anchor_obj_count x 4
        loc_gt = tf.reshape(loc_gt, (-1, 1))
        loc_pred = tf.reshape(loc_pred, (-1, 1))
        loc_diff = tf.abs(loc_gt - loc_pred)
        cond = tf.less(loc_diff, beta)
        smooth_l1_loss = tf.where(cond, 0.5 * loc_diff**2 / beta, loc_diff - 0.5 * beta)
        smooth_l1_loss = tf.reduce_sum(smooth_l1_loss) / tf.cast(anchor_obj_count, tf.float32)
        return smooth_l1_loss

    def forward(self, data, state):
        cls_gt, x1_gt, y1_gt, w_gt, h_gt, cls_pred, loc_pred = data
        local_batch_size = state["local_batch_size"]
        focal_loss = []
        l1_loss = []
        total_loss = []
        for idx in range(local_batch_size):
            cls_gt_example = cls_gt[idx]
            x1_gt_example = x1_gt[idx]
            y1_gt_example = y1_gt[idx]
            w_gt_example = w_gt[idx]
            h_gt_example = h_gt[idx]
            loc_gt_example = tf.transpose(tf.stack([x1_gt_example, y1_gt_example, w_gt_example, h_gt_example]))
            cls_pred_example = cls_pred[idx]
            loc_pred_example = loc_pred[idx]
            focal_loss_example, anchor_obj_idx = self.focal_loss(cls_gt_example, cls_pred_example)
            smooth_l1_loss_example = self.smooth_l1(loc_gt_example, loc_pred_example, anchor_obj_idx)
            focal_loss.append(focal_loss_example)
            l1_loss.append(smooth_l1_loss_example)
        focal_loss = tf.stack(focal_loss)
        l1_loss = tf.stack(l1_loss)
        return focal_loss, l1_loss


def get_estimator(data_path=None, model_dir=tempfile.mkdtemp(), batch_size=8):
    #prepare dataset
    train_csv, val_csv, path = load_data(path=data_path)
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "retinanet_coco"),
        train_data=train_csv,
        validation_data=val_csv,
        ops=[
            ImageReader(inputs="image", parent_path=path, outputs="image"),
            String2List(inputs=["x1", "y1", "width", "height", "obj_label"],
                        outputs=["x1", "y1", "width", "height", "obj_label"]),
            ResizeImageAndBbox(target_size=(512, 512),
                               keep_ratio=True,
                               inputs=["image", "x1", "y1", "width", "height"],
                               outputs=["image", "x1", "y1", "width", "height"]),
            GenerateTarget(inputs=("obj_label", "x1", "y1", "width", "height"),
                           outputs=("cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt")),
            TypeConverter(target_type='int32', inputs=["id", "cls_gt"], outputs=["id", "cls_gt"]),
            TypeConverter(target_type='float32',
                          inputs=["x1_gt", "y1_gt", "w_gt", "h_gt"],
                          outputs=["x1_gt", "y1_gt", "w_gt", "h_gt"])
        ],
        compression="GZIP",
        write_feature=[
            "image", "id", "cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt", "obj_label", "x1", "y1", "width", "height"
        ])
    # prepare pipeline
    pipeline = fe.Pipeline(
        batch_size=batch_size,
        data=writer,
        ops=[
            Rescale(inputs="image", outputs="image"),
            Pad(padded_shape=[190],
                inputs=["x1_gt", "y1_gt", "w_gt", "h_gt", "obj_label", "x1", "y1", "width", "height"],
                outputs=["x1_gt", "y1_gt", "w_gt", "h_gt", "obj_label", "x1", "y1", "width", "height"])
        ])
    # prepare network
    retinanet = fe.build(model_def=lambda: RetinaNet(input_shape=(512, 512, 3), num_classes=90),
                         model_name="retinanet",
                         optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                         loss_name="total_loss")
    uncertainty_loss = fe.build(model_def=lambda: UncertaintyLoss(num_losses=2),
                                model_name="uncertainty_loss",
                                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                                loss_name="total_loss")

    network = fe.Network(ops=[
        ModelOp(inputs="image", model=retinanet, outputs=["cls_pred", "loc_pred"]),
        RetinaLoss(inputs=("cls_gt", "x1_gt", "y1_gt", "w_gt", "h_gt", "cls_pred", "loc_pred"),
                   outputs=("focal_loss", "l1_loss")),
        ModelOp(inputs=["focal_loss", "l1_loss"], model=uncertainty_loss, outputs="total_loss"),
        Loss(inputs="total_loss", outputs="total_loss"),
        PredictBox(inputs=["cls_pred", "loc_pred", "obj_label", "x1", "y1", "width", "height"],
                   outputs=("pred", "gt"),
                   mode="eval")
    ])
    # prepare estimator
    estimator = fe.Estimator(
        network=network,
        pipeline=pipeline,
        epochs=15,
        traces=[
            MeanAvgPrecision(90, (512, 512, 3), 'pred', 'gt', output_name='mAP'),
            ModelSaver(model_name="retinanet", save_dir=model_dir, save_best='mAP'),
        ])
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
