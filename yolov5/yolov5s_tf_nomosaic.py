import math
import pdb
import random
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import tensorflow as tf
from albumentations import BboxParams
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import CenterCrop, HorizontalFlip, LongestMaxSize, PadIfNeeded
from fastestimator.op.numpyop.univariate import ReadImage, ToArray
from fastestimator.op.tensorop import Average, TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.schedule import EpochScheduler, cosine_decay
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver, RestoreWizard
from fastestimator.trace.metric import MeanAveragePrecision
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant


class HSVAugment(NumpyOp):
    def __init__(self, inputs, outputs, mode="train", hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v

    def forward(self, data, state):
        img = data
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img


class CategoryID2ClassID(NumpyOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(category, list(range(80)))}

    def forward(self, data, state):
        if data.size > 0:
            classes = np.array([self.mapping[int(x)] for x in data[:, -1]], dtype="float32")
            data[:, -1] = classes
        else:
            data = np.zeros(shape=(1, 5), dtype="float32")
        return data


class COCO2Yolo(NumpyOp):
    def __init__(self, inputs, outputs, width, height, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.width = width
        self.height = height

    def forward(self, data, state):
        x1, y1, w, h, label = np.split(data, 5, axis=1)
        xc, yc = (x1 + w / 2) / self.width, (y1 + h / 2) / self.height
        w_rario, h_ratio = w / self.width, h / self.height
        return np.concatenate([xc, yc, w_rario, h_ratio, label], axis=1)


class GTBox(NumpyOp):
    def __init__(self, inputs, outputs, width, height, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.width = width
        self.height = height
        self.anchor_s = [(10, 13), (16, 30), (33, 23)]
        self.anchor_m = [(30, 61), (62, 45), (59, 119)]
        self.anchor_l = [(116, 90), (156, 198), (373, 326)]

    def forward(self, data, state):
        gt_sbbox = self._generate_target(data, anchors=self.anchor_s, feature_size=80)
        gt_mbbox = self._generate_target(data, anchors=self.anchor_m, feature_size=40)
        gt_lbbox = self._generate_target(data, anchors=self.anchor_l, feature_size=20)
        return gt_sbbox, gt_mbbox, gt_lbbox

    def get_anchor_boxes(self, object_boxes, anchors, feature_size):
        anchor_boxes = []
        for (anchor_w, anchor_h) in anchors:
            xc = (np.floor(object_boxes[:, 0:1] * feature_size) + 0.5) / feature_size
            yc = (np.floor(object_boxes[:, 1:2] * feature_size) + 0.5) / feature_size
            anchor_box = np.concatenate(
                [xc, yc, anchor_w / self.width * np.ones_like(xc), anchor_h / self.height * np.ones_like(xc)], axis=1)
            anchor_boxes.append(anchor_box)
        return anchor_boxes

    def _generate_target(self, bbox, anchors, feature_size):
        object_boxes, label = bbox[:, :-1], bbox[:, -1]
        gt_bbox = np.zeros((feature_size, feature_size, 3, 6), dtype="float32")
        num_objects = object_boxes.shape[0]
        anchor_boxes = self.get_anchor_boxes(object_boxes, anchors, feature_size)
        ious = np.stack([np.diag(self._get_iou(object_boxes, anchor_box)) for anchor_box in anchor_boxes])
        matched = ious > 0.3  # any IOU >0.3 centered at object center in feature map gets a match
        matched[np.argmax(ious, 0), range(num_objects)] = True  # the highest IOU of each object gets a match
        for i in range(len(anchor_boxes)):
            for j in range(num_objects):
                if matched[i, j]:
                    anchor_xc_coord = int(anchor_boxes[i][j][0] * feature_size)
                    anchor_yc_coord = int(anchor_boxes[i][j][1] * feature_size)
                    gt_bbox[anchor_yc_coord, anchor_xc_coord,
                            i][0:2] = (object_boxes[j][0:2] - anchor_boxes[i][j][0:2] +
                                       1 / feature_size / 2) * feature_size  #center w.r.t the gridcell
                    gt_bbox[anchor_yc_coord, anchor_xc_coord, i][2:4] = object_boxes[j][2:4]  #width and height
                    gt_bbox[anchor_yc_coord, anchor_xc_coord, i][4] = 1.0
                    gt_bbox[anchor_yc_coord, anchor_xc_coord, i][5] = label[j]
        return gt_bbox

    @staticmethod
    def _get_iou(boxes1, boxes2):
        """Computes the value of intersection over union (IoU) of two array of boxes.
        Args:
            box1 (array): first boxes in N x 4
            box2 (array): second box in M x 4
        Returns:
            float: IoU value in N x M
        """
        x1c, y1c, w1, h1 = np.split(boxes1, 4, axis=1)
        x2c, y2c, w2, h2 = np.split(boxes2, 4, axis=1)
        x11, x12, y11, y12 = x1c - w1 / 2, x1c + w1 / 2, y1c - h1 / 2, y1c + h1 / 2
        x21, x22, y21, y22 = x2c - w2 / 2, x2c + w2 / 2, y2c - h2 / 2, y2c + h2 / 2
        xmin = np.maximum(x11, np.transpose(x21))
        ymin = np.maximum(y11, np.transpose(y21))
        xmax = np.minimum(x12, np.transpose(x22))
        ymax = np.minimum(y12, np.transpose(y22))
        inter_area = np.maximum((xmax - xmin + 1), 0) * np.maximum((ymax - ymin + 1), 0)
        area1 = (w1 + 1) * (h1 + 1)
        area2 = (w2 + 1) * (h2 + 1)
        iou = inter_area / (area1 + area2.T - inter_area)
        return iou


def conv_block(x, c, k=1, s=1):
    x = layers.Conv2D(filters=c, kernel_size=k, strides=s, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.97)(x)
    x = tf.nn.silu(x)
    return x


def bottleneck(x, c, k=1, shortcut=True):
    out = conv_block(x, c=c, k=1)
    out = conv_block(out, c=c, k=3)
    if shortcut and c == x.shape[-1]:
        out = out + x
    return out


def csp_bottleneck_conv3(x, c, n=1, shortcut=True):
    out1 = conv_block(x, c=c // 2)
    for _ in range(n):
        out1 = bottleneck(out1, c=c // 2, shortcut=shortcut)
    out2 = conv_block(x, c=c // 2)
    out = tf.concat([out1, out2], axis=-1)
    out = conv_block(out, c=c)
    return out


def spatial_pyramid_pooling(x, c, k=(5, 9, 13)):
    input_c = x.shape[-1]
    x = conv_block(x, c=input_c // 2)
    x = tf.concat([x] + [layers.MaxPool2D(pool_size=p, strides=1, padding='same')(x) for p in k], axis=-1)
    x = conv_block(x, c=c)
    return x


def yolov5(input_shape, num_classes, strides=(8, 16, 32)):
    inp = layers.Input(shape=input_shape)
    x = tf.concat([inp[:, ::2, ::2, :], inp[:, 1::2, ::2, :], inp[:, ::2, 1::2, :], inp[:, 1::2, 1::2, :]], axis=-1)
    x = conv_block(x, c=32, k=3)
    x = conv_block(x, c=64, k=3, s=2)
    x = csp_bottleneck_conv3(x, c=64)
    x = conv_block(x, c=128, k=3, s=2)
    x_4 = csp_bottleneck_conv3(x, c=128, n=3)
    x = conv_block(x_4, c=256, k=3, s=2)
    x_6 = csp_bottleneck_conv3(x, c=256, n=3)
    x = conv_block(x_6, c=512, k=3, s=2)
    x = spatial_pyramid_pooling(x, c=512)
    x = csp_bottleneck_conv3(x, 512, shortcut=False)
    x_10 = conv_block(x, 256)
    x = layers.UpSampling2D()(x_10)
    x = tf.concat([x, x_6], axis=-1)
    x = csp_bottleneck_conv3(x, 256, shortcut=False)
    x_14 = conv_block(x, 128)
    x = layers.UpSampling2D()(x_14)
    x = tf.concat([x, x_4], axis=-1)
    x_17 = csp_bottleneck_conv3(x, 128, shortcut=False)
    x = conv_block(x_17, 128, 3, 2)
    x = tf.concat([x, x_14], axis=-1)
    x_20 = csp_bottleneck_conv3(x, 256, shortcut=False)
    x = conv_block(x_20, 256, 3, 2)
    x = tf.concat([x, x_10], axis=-1)
    x_23 = csp_bottleneck_conv3(x, 512, shortcut=False)
    # initialize the bias for the final layer
    biases = []
    for stride, in_channel in zip(strides, (128, 256, 512)):
        bias = np.random.uniform(low=-(1 / in_channel)**0.5, high=(1 / in_channel)**0.5, size=(3, num_classes + 5))
        bias[:, 4] += math.log(8 / (640 / stride)**2)  # obj (8 objects per 640 image)
        bias[:, 5:] += math.log(0.6 / (num_classes - 0.99))  # cls
        biases.append(bias.flatten())
    out_17 = layers.Conv2D((num_classes + 5) * 3, 1, bias_initializer=Constant(biases[0]))(x_17)
    out_17 = layers.Reshape((out_17.shape[1], out_17.shape[2], 3, num_classes + 5))(out_17)
    out_20 = layers.Conv2D((num_classes + 5) * 3, 1, bias_initializer=Constant(biases[0]))(x_20)
    out_20 = layers.Reshape((out_20.shape[1], out_20.shape[2], 3, num_classes + 5))(out_20)
    out_23 = layers.Conv2D((num_classes + 5) * 3, 1, bias_initializer=Constant(biases[0]))(x_23)
    out_23 = layers.Reshape((out_23.shape[1], out_23.shape[2], 3, num_classes + 5))(out_23)  # B, h/32, w/32, 3, 85
    return tf.keras.Model(inputs=inp, outputs=[out_17, out_20, out_23])


class ComputeLoss(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_conf = tf.losses.BinaryCrossentropy(from_logits=True, reduction='sum')
        self.loss_bbox = tf.losses.MeanSquaredError(reduction='sum')
        self.loss_cls = tf.losses.BinaryCrossentropy(from_logits=True, reduction='sum')
        self.bbox_loss_multi = 5
        self.conf_loss_multi = 0.5

    def forward(self, data, state):
        conv_bbox, gt_bbox = data
        batch_size = gt_bbox.shape[0]
        bbox_loss, conf_loss, cls_loss = [], [], []
        for idx in range(batch_size):
            conv_bbox_single, gt_bbox_single = conv_bbox[idx], gt_bbox[idx]
            has_obj = gt_bbox_single[:, :, :, 4] == 1.0
            conv_bbox_single_obj, gt_bbox_single_obj = conv_bbox_single[has_obj], gt_bbox_single[has_obj]
            num_obj = tf.cast(tf.shape(conv_bbox_single_obj)[0], tf.float32)
            conf_loss.append(self.get_conf_loss(conv_bbox_single, gt_bbox_single) / num_obj)
            bbox_loss.append(self.get_bbox_loss(conv_bbox_single_obj, gt_bbox_single_obj) / num_obj)
            cls_loss.append(self.get_cls_loss(conv_bbox_single_obj, gt_bbox_single_obj) / num_obj)
        return self.bbox_loss_multi  * tf.reduce_mean(bbox_loss), self.conf_loss_multi * tf.reduce_mean(conf_loss), tf.reduce_mean(cls_loss)

    def get_conf_loss(self, pred, gt):
        pred_conf, gt_conf = tf.reshape(pred[:, :, :, 4], (-1, 1)), tf.reshape(gt[:, :, :, 4], (-1, 1))
        conf_loss = self.loss_conf(gt_conf, pred_conf)
        return conf_loss

    def get_bbox_loss(self, pred, gt):
        xy_pred, xy_gt = tf.sigmoid(pred[:, 0:2]), gt[:, 0:2]
        wh_pred, wh_gt = tf.sigmoid(pred[:, 2:4]), gt[:, 2:4]
        bbox_loss_xy = self.loss_bbox(xy_gt, xy_pred)
        bbox_loss_wh = self.loss_bbox(tf.sqrt(wh_gt), wh_pred)
        return bbox_loss_xy + bbox_loss_wh

    def get_cls_loss(self, pred, gt):
        cls_pred, cls_gt = pred[:, 5:], tf.cast(gt[:, 5], tf.int32)
        cls_gt = tf.one_hot(cls_gt, cls_pred.shape[-1])
        cls_pred, cls_gt = tf.reshape(cls_pred, (-1, 1)), tf.reshape(cls_gt, (-1, 1))
        cls_loss = self.loss_cls(cls_gt, cls_pred)
        return cls_loss


class CombineLoss(TensorOp):
    def forward(self, data, state):
        return tf.reduce_sum(data)


class Rescale(TensorOp):
    def forward(self, data, state):
        return data / 255


class PredictBox(TensorOp):
    def __init__(self,
                 inputs,
                 outputs,
                 mode,
                 width,
                 height,
                 select_top_k=1000,
                 nms_max_outputs=100,
                 score_threshold=0.05):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.width = width
        self.height = height
        self.select_top_k = select_top_k
        self.nms_max_outputs = nms_max_outputs
        self.score_threshold = score_threshold
        self.strides = [8, 16, 32]
        self.num_anchor = 3
        self.grids = self.create_grid(self.strides, self.num_anchor)

    def create_grid(self, strides, num_anchor):
        grids = []
        for stride in strides:
            x_coor = [stride * i for i in range(self.width // stride)]
            y_coor = [stride * i for i in range(self.height // stride)]
            xx, yy = np.meshgrid(x_coor, y_coor)
            xx, yy = np.float32(xx), np.float32(yy)
            xx, yy = np.stack([xx] * num_anchor, axis=-1), np.stack([yy] * num_anchor, axis=-1)
            grids.append((xx, yy))
        return grids

    def forward(self, data, state):
        conv_sbbox, conv_mbbox, conv_lbbox = data
        batch_size = conv_sbbox.shape[0]
        final_results = []
        for idx in range(batch_size):
            conv_bboxes = [conv_sbbox[idx], conv_mbbox[idx], conv_lbbox[idx]]
            selected_bboxes = []
            for conv_bbox, (xx, yy), stride in zip(conv_bboxes, self.grids, self.strides):
                # convert prediction to absolute scale
                conv_bbox = tf.sigmoid(conv_bbox)
                width_abs = conv_bbox[:, :, :, 2] * self.width
                height_abs = conv_bbox[:, :, :, 3] * self.height
                x1_abs = conv_bbox[:, :, :, 0] * stride + xx - width_abs / 2
                y1_abs = conv_bbox[:, :, :, 1] * stride + yy - height_abs / 2
                x2_abs = x1_abs + width_abs
                y2_abs = y1_abs + height_abs
                obj_score = conv_bbox[:, :, :, 4]
                label = tf.cast(tf.argmax(conv_bbox[:, :, :, 5:], axis=-1), tf.float32)
                label_score = tf.reduce_max(conv_bbox[:, :, :, 5:], axis=-1)
                all_conv_bboxes = tf.stack([y1_abs, x1_abs, y2_abs, x2_abs, obj_score, label, label_score], axis=-1)
                all_conv_bboxes = tf.reshape(all_conv_bboxes, (-1, 7))
                # select the top 1k bboxes
                selected_idx = tf.math.top_k(all_conv_bboxes[:, 4],
                                             tf.minimum(self.select_top_k, tf.shape(all_conv_bboxes)[0])).indices
                selected_bboxes.append(tf.gather(all_conv_bboxes, selected_idx))
            selected_bboxes = tf.concat(selected_bboxes, axis=0)
            # nms
            nms_keep = tf.image.non_max_suppression(selected_bboxes[:, :4],
                                                    selected_bboxes[:, 4],
                                                    self.nms_max_outputs,
                                                    score_threshold=self.score_threshold)
            selected_bboxes = tf.gather(selected_bboxes, nms_keep)
            # clip bounding boxes to image size
            y1_abs = tf.clip_by_value(selected_bboxes[:, 0], 0, self.height)
            x1_abs = tf.clip_by_value(selected_bboxes[:, 1], 0, self.width)
            height_abs = tf.clip_by_value(selected_bboxes[:, 2] - y1_abs, 0, self.height - y1_abs)
            width_abs = tf.clip_by_value(selected_bboxes[:, 3] - x1_abs, 0, self.width - x1_abs)
            labels, labels_score = selected_bboxes[:, 5], selected_bboxes[:, 6]
            # final output: [x1, y1, w, h, label, label_score, select_or_not]
            results_single = [x1_abs, y1_abs, width_abs, height_abs, labels, labels_score, tf.ones_like(x1_abs)]
            results_single = tf.stack(results_single, axis=-1)
            # pad 0 to other rows to improve performance
            results_single = tf.pad(results_single, [(0, self.nms_max_outputs - tf.shape(results_single)[0]), (0, 0)])
            final_results.append(results_single)
        final_results = tf.stack(final_results)
        return final_results


def lr_fn(step):
    if step < 2000:
        lr = (0.01 - 0.0002) / 2000 * step + 0.0002
    elif step < 1832 * 200:
        lr = 0.01
    elif step < 1832 * 250:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


def get_estimator(data_dir="/data/data/public/COCO2017/",
                  model_dir=tempfile.mkdtemp(),
                  restore_dir=tempfile.mkdtemp(),
                  epochs=300):
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        batch_size=64,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            LongestMaxSize(max_size=640, image_in="image", bbox_in="bbox", bbox_params=BboxParams("coco",
                                                                                                  min_area=1.0)),
            PadIfNeeded(min_height=640,
                        min_width=640,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            Sometimes(
                HorizontalFlip(image_in="image",
                               bbox_in="bbox",
                               bbox_params=BboxParams("coco", min_area=1.0),
                               mode="train")),
            HSVAugment(inputs="image", outputs="image", mode="train"),
            ToArray(inputs="bbox", outputs="bbox", dtype="float32"),
            CategoryID2ClassID(inputs="bbox", outputs="bbox"),
            COCO2Yolo(inputs="bbox", outputs="bbox_yolo", width=640, height=640),
            GTBox(inputs="bbox_yolo", outputs=("gt_sbbox", "gt_mbbox", "gt_lbbox"), width=640, height=640),
        ],
        pad_value=0)
    model = fe.build(lambda: yolov5(input_shape=(640, 640, 3), num_classes=80),
                     optimizer_fn=lambda: tf.optimizers.SGD(momentum=0.9))
    network = fe.Network(ops=[
        Rescale(inputs="image", outputs="image"),
        ModelOp(model=model, inputs="image", outputs=("conv_sbbox", "conv_mbbox", "conv_lbbox")),
        ComputeLoss(inputs=("conv_sbbox", "gt_sbbox"), outputs=("sbbox_loss", "sconf_loss", "scls_loss")),
        ComputeLoss(inputs=("conv_mbbox", "gt_mbbox"), outputs=("mbbox_loss", "mconf_loss", "mcls_loss")),
        ComputeLoss(inputs=("conv_lbbox", "gt_lbbox"), outputs=("lbbox_loss", "lconf_loss", "lcls_loss")),
        Average(inputs=("sbbox_loss", "mbbox_loss", "lbbox_loss"), outputs="bbox_loss"),
        Average(inputs=("sconf_loss", "mconf_loss", "lconf_loss"), outputs="conf_loss"),
        Average(inputs=("scls_loss", "mcls_loss", "lcls_loss"), outputs="cls_loss"),
        CombineLoss(inputs=("bbox_loss", "conf_loss", "cls_loss"), outputs="total_loss"),
        PredictBox(
            width=640, height=640, inputs=("conv_sbbox", "conv_mbbox", "conv_lbbox"), outputs="box_pred", mode="eval"),
        UpdateOp(model=model, loss_name="total_loss")
    ])
    traces = [
        LRScheduler(model=model, lr_fn=lr_fn),
        BestModelSaver(model=model, save_dir=model_dir, metric='mAP', save_best_mode="max"),
        MeanAveragePrecision(num_classes=80, true_key='bbox', pred_key='box_pred', mode="eval"),
        RestoreWizard(directory=restore_dir)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=["bbox_loss", "conf_loss", "cls_loss"])
    return estimator
