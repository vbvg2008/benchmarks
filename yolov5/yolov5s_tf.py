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
from torch.utils.data import Dataset


# This dataset selects 4 images and its bboxes
class PreMosaicDataset(Dataset):
    def __init__(self, mscoco_ds):
        self.mscoco_ds = mscoco_ds

    def __len__(self):
        return len(self.mscoco_ds)

    def __getitem__(self, idx):
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        samples = [self.mscoco_ds[i] for i in indices]
        return {
            "image1": samples[0]["image"],
            "bbox1": samples[0]["bbox"],
            "image2": samples[1]["image"],
            "bbox2": samples[1]["bbox"],
            "image3": samples[2]["image"],
            "bbox3": samples[2]["bbox"],
            "image4": samples[3]["image"],
            "bbox4": samples[3]["bbox"]
        }


class CombineMosaic(NumpyOp):
    def forward(self, data, state):
        image1, image2, image3, image4, bbox1, bbox2, bbox3, bbox4 = data
        images = [image1, image2, image3, image4]
        bboxes = [bbox1, bbox2, bbox3, bbox4]
        images_new, boxes_new = self._combine_images_boxes(images, bboxes)
        return images_new, boxes_new

    def _combine_images_boxes(self, images, bboxes):
        s = 640
        yc, xc = int(random.uniform(320, 960)), int(random.uniform(320, 960))
        images_new = np.full((1280, 1280, 3), fill_value=114, dtype=np.uint8)
        bboxes_new = []
        for idx, (image, bbox) in enumerate(zip(images, bboxes)):
            h, w = image.shape[0], image.shape[1]
            # place img in img4
            if idx == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif idx == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif idx == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif idx == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            images_new[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            for x1, y1, bw, bh, label in bbox:
                x1_new = np.clip(x1 + padw, x1a, x2a)
                y1_new = np.clip(y1 + padh, y1a, y2a)
                x2_new = np.clip(x1 + padw + bw, x1a, x2a)
                y2_new = np.clip(y1 + padh + bh, y1a, y2a)
                bw_new = x2_new - x1_new
                bh_new = y2_new - y1_new
                if bw_new * bh_new > 1:
                    bboxes_new.append((x1_new, y1_new, bw_new, bh_new, label))
        return images_new, bboxes_new


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


class GTBox(NumpyOp):
    def __init__(self, inputs, outputs, width, height, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.width = width
        self.height = height
        self.anchor_s = self.get_anchor_boxes([(10, 13), (16, 30), (33, 23)], 80)
        self.anchor_m = self.get_anchor_boxes([(30, 61), (62, 45), (59, 119)], 40)
        self.anchor_l = self.get_anchor_boxes([(116, 90), (156, 198), (373, 326)], 20)

    def forward(self, data, state):
        bbox = data[np.sum(data, 1) > 0]
        if bbox.size > 0:
            ious_s, object_box_s, label = self._prepare_boxes(data, self.anchor_s, 80)
            ious_m, object_box_m, _ = self._prepare_boxes(data, self.anchor_m, 40)
            ious_l, object_box_l, _ = self._prepare_boxes(data, self.anchor_l, 20)
            matched_s, matched_m, matched_l = self._match_boxes(ious_s, ious_m, ious_l)
            gt_sbbox = self._generate_target(matched_s, object_box_s, self.anchor_s, feature_size=80, label=label)
            gt_mbbox = self._generate_target(matched_m, object_box_m, self.anchor_m, feature_size=40, label=label)
            gt_lbbox = self._generate_target(matched_l, object_box_l, self.anchor_l, feature_size=20, label=label)
        else:
            gt_sbbox = np.zeros((80, 80, 3, 6), dtype="float32")
            gt_mbbox = np.zeros((40, 40, 3, 6), dtype="float32")
            gt_lbbox = np.zeros((20, 20, 3, 6), dtype="float32")
        return gt_sbbox, gt_mbbox, gt_lbbox

    def get_anchor_boxes(self, anchor_wh, feature_size):
        xc_coor, yc_coor = [i + 0.5 for i in range(feature_size)], [i + 0.5 for i in range(feature_size)]
        xc, yc = np.meshgrid(xc_coor, yc_coor)
        xc, yc = np.float32(xc), np.float32(yc)
        xc, yc = np.stack([xc] * len(anchor_wh), axis=-1), np.stack([yc] * len(anchor_wh), axis=-1)
        anchor_wh = self.img2feature(np.array(anchor_wh, dtype="float32").reshape(1, 1, 3, 2), feature_size, 640)
        anchor_wh = np.tile(anchor_wh, [feature_size, feature_size, 1, 1])
        anchor_width, anchor_height = anchor_wh[..., 0], anchor_wh[..., 1]
        x1, y1 = xc - anchor_width / 2, yc - anchor_height / 2
        anchor_boxes = np.stack([x1, y1, anchor_width, anchor_height], axis=-1)
        return np.reshape(anchor_boxes, (-1, 4))

    def _match_boxes(self, ious_s, ious_m, ious_l):
        num_anchors = np.cumsum([ious_s.shape[1], ious_m.shape[1], ious_l.shape[1]])
        ious = np.concatenate([ious_s, ious_m, ious_l], axis=1)
        matched = np.zeros_like(ious)
        matched[ious > 0.5] = -1  # anything > 0.5 IOU is marked -1
        matched = np.min(matched, axis=0)
        best_match_boxes = np.argmax(ious, 1)
        for obj_idx, box_idx in enumerate(best_match_boxes):
            matched[box_idx] = obj_idx + 1  #object index starting from 1
        return matched[:num_anchors[0]], matched[num_anchors[0]:num_anchors[1]], matched[num_anchors[1]:]

    @staticmethod
    def img2feature(x, feature_size, image_size):
        return x / image_size * feature_size

    def _prepare_boxes(self, bbox, anchors, feature_size):
        object_box, label = bbox[:, :-1], bbox[:, -1]
        object_box = self.img2feature(object_box, feature_size, 640)  #x1, y1, w, h
        ious = self._get_iou(object_box, anchors)
        return ious, object_box, label

    def _generate_target(self, matched, object_box, anchor_boxes, feature_size, label):
        gt_bbox = np.zeros((feature_size, feature_size, 3, 6), dtype="float32")
        matched = matched.reshape(feature_size, feature_size, 3)
        anchor_boxes = anchor_boxes.reshape(feature_size, feature_size, 3, 4)
        y_locs, x_locs, a_locs = np.where(matched != 0)
        for y_loc, x_loc, a_loc in zip(y_locs, x_locs, a_locs):
            if matched[y_loc, x_loc, a_loc] == -1:
                gt_bbox[y_loc, x_loc, a_loc][4] = -1.0  # ignore these boxes later
            elif matched[y_loc, x_loc, a_loc] > 0:
                obj_idx = int(matched[y_loc, x_loc, a_loc]) - 1
                gt_bbox[y_loc, x_loc, a_loc][0:2] = (object_box[obj_idx][0:2] + object_box[obj_idx][2:4] / 2) % 1
                gt_bbox[y_loc, x_loc,
                        a_loc][2:4] = np.log(object_box[obj_idx][2:4] / anchor_boxes[y_loc, x_loc, a_loc][2:4])
                gt_bbox[y_loc, x_loc, a_loc][4] = 1.0
                gt_bbox[y_loc, x_loc, a_loc][5] = label[obj_idx]
        return gt_bbox

    @staticmethod
    def _get_iou(boxes1, boxes2):
        x11, y11, w1, h1 = np.split(boxes1, 4, axis=1)
        x21, y21, w2, h2 = np.split(boxes2, 4, axis=1)
        x12, y12 = x11 + w1, y11 + h1
        x22, y22 = x21 + w2, y21 + h2
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
        self.bbox_loss_multi = 10
        self.conf_loss_multi = 0.05
        self.cls_loss_multi = 1.0

    def forward(self, data, state):
        conv_bbox, gt_bbox = data
        batch_size = gt_bbox.shape[0]
        bbox_loss, conf_loss, cls_loss = tf.zeros(()), tf.zeros(()), tf.zeros(())
        for idx in range(batch_size):
            conv_bbox_single, gt_bbox_single = conv_bbox[idx], gt_bbox[idx]
            consider_conf = gt_bbox_single[:, :, :, 4] > -1.0
            has_obj = gt_bbox_single[:, :, :, 4] == 1.0
            conf_loss += self.get_conf_loss(conv_bbox_single[consider_conf], gt_bbox_single[consider_conf])
            conv_bbox_single_obj, gt_bbox_single_obj = conv_bbox_single[has_obj], gt_bbox_single[has_obj]
            num_obj = tf.cast(tf.shape(conv_bbox_single_obj)[0], tf.float32)
            if num_obj > 0:
                bbox_loss += self.get_bbox_loss(conv_bbox_single_obj, gt_bbox_single_obj) / num_obj
                cls_loss += self.get_cls_loss(conv_bbox_single_obj, gt_bbox_single_obj) / num_obj
        final_bbox_loss = self.bbox_loss_multi * bbox_loss / tf.cast(batch_size, tf.float32)
        final_conf_loss = self.conf_loss_multi * conf_loss / tf.cast(batch_size, tf.float32)
        final_cls_loss = self.cls_loss_multi * cls_loss / tf.cast(batch_size, tf.float32)
        return final_bbox_loss, final_conf_loss, final_cls_loss

    def get_conf_loss(self, pred, gt):
        pred_conf, gt_conf = tf.reshape(pred[:, 4], (-1, 1)), tf.reshape(gt[:, 4], (-1, 1))
        conf_loss = self.loss_conf(gt_conf, pred_conf)
        return conf_loss

    def get_bbox_loss(self, pred, gt):
        xy_pred, xy_gt = tf.sigmoid(pred[:, 0:2]), gt[:, 0:2]
        wh_pred, wh_gt = pred[:, 2:4], gt[:, 2:4]
        bbox_loss_xy = self.loss_bbox(xy_gt, xy_pred)
        bbox_loss_wh = self.loss_bbox(wh_gt, wh_pred)
        return bbox_loss_xy + bbox_loss_wh / 100

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
        anchor_s = [(10, 13), (16, 30), (33, 23)]
        anchor_m = [(30, 61), (62, 45), (59, 119)]
        anchor_l = [(116, 90), (156, 198), (373, 326)]
        self.anchors = self.create_anchor(anchor_s, anchor_m, anchor_l, self.strides)

    def create_anchor(self, anchor_s, anchor_m, anchor_l, strides):
        anchors = []
        for anchor, stride in zip([anchor_s, anchor_m, anchor_l], strides):
            feature_size_x, feature_size_y = self.width // stride, self.height // stride
            anchor = np.array(anchor, dtype="float32").reshape((1, 1, 3, 2))
            anchor = np.tile(anchor, [feature_size_y, feature_size_x, 1, 1])
            anchors.append(tf.convert_to_tensor(anchor))
        return anchors

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
            for conv_bbox, (xx, yy), stride, anchor in zip(conv_bboxes, self.grids, self.strides, self.anchors):
                # convert prediction to absolute scale
                width_abs = tf.exp(conv_bbox[:, :, :, 2]) * anchor[:, :, :, 0]
                height_abs = tf.exp(conv_bbox[:, :, :, 3]) * anchor[:, :, :, 1]
                x1_abs = tf.sigmoid(conv_bbox[:, :, :, 0]) * stride + xx - width_abs / 2
                y1_abs = tf.sigmoid(conv_bbox[:, :, :, 1]) * stride + yy - height_abs / 2
                x2_abs = x1_abs + width_abs
                y2_abs = y1_abs + height_abs
                obj_score = tf.sigmoid(conv_bbox[:, :, :, 4])
                label = tf.cast(tf.argmax(conv_bbox[:, :, :, 5:], axis=-1), tf.float32)
                label_score = tf.reduce_max(tf.sigmoid(conv_bbox[:, :, :, 5:]), axis=-1)
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
        lr = (0.1 - 0.002) / 2000 * step + 0.002
    elif step < 1833 * 200:
        lr = 0.1
    elif step < 1833 * 250:
        lr = 0.01
    else:
        lr = 0.001
    return lr


def get_estimator(data_dir="/data/data/public/COCO2017/",
                  model_dir=tempfile.mkdtemp(),
                  restore_dir=tempfile.mkdtemp(),
                  epochs=300,
                  batch_size=64):
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir)
    train_ds = PreMosaicDataset(mscoco_ds=train_ds)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs=("image1", "image2", "image3", "image4"),
                      outputs=("image1", "image2", "image3", "image4"),
                      mode="train"),
            ReadImage(inputs="image", outputs="image", mode="eval"),
            LongestMaxSize(max_size=640,
                           image_in="image1",
                           bbox_in="bbox1",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image2",
                           bbox_in="bbox2",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image3",
                           bbox_in="bbox3",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image4",
                           bbox_in="bbox4",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="train"),
            LongestMaxSize(max_size=640,
                           image_in="image",
                           bbox_in="bbox",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="eval"),
            PadIfNeeded(min_height=640,
                        min_width=640,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="eval",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            CombineMosaic(inputs=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4"),
                          outputs=("image", "bbox"),
                          mode="train"),
            CenterCrop(height=640,
                       width=640,
                       image_in="image",
                       bbox_in="bbox",
                       bbox_params=BboxParams("coco", min_area=1.0),
                       mode="train"),
            Sometimes(
                HorizontalFlip(image_in="image",
                               bbox_in="bbox",
                               bbox_params=BboxParams("coco", min_area=1.0),
                               mode="train")),
            HSVAugment(inputs="image", outputs="image", mode="train"),
            ToArray(inputs="bbox", outputs="bbox", dtype="float32"),
            CategoryID2ClassID(inputs="bbox", outputs="bbox"),
            GTBox(inputs="bbox", outputs=("gt_sbbox", "gt_mbbox", "gt_lbbox"), width=640, height=640),
            Delete(keys=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4"), mode="train")
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
