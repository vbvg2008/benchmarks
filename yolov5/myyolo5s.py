import json
import math
import pdb
import random
import tempfile

import cv2
import fastestimator as fe
import numpy as np
import torch
import torch.nn as nn
import torchvision
from albumentations import BboxParams
from fastestimator.dataset.data import mscoco
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, RandomCrop, \
    RandomScale, Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Normalize, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import MeanAveragePrecision
from torch.utils.data import Dataset


# Reuseable convolution
class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Standard bottleneck
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):  # ch_in, ch_out, shortcut
        super().__init__()
        self.cv1 = ConvBlock(c1, c2, 1)
        self.cv2 = ConvBlock(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        if self.add:
            out = out + x
        return out


# CSP Bottleneck with 3 convolutions
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):  # ch_in, ch_out, num_repeat, shortcut
        super().__init__()
        self.cv1 = ConvBlock(c1, c2 // 2)
        self.cv2 = ConvBlock(c1, c2 // 2)
        self.m = nn.Sequential(*[Bottleneck(c2 // 2, c2 // 2, shortcut) for _ in range(n)])
        self.cv3 = ConvBlock(c2, c2)

    def forward(self, x):
        out1 = self.cv1(x)
        out1 = self.m(out1)
        out2 = self.cv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.cv3(out)
        return out


# Focus wh information into c-space
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = ConvBlock(c1 * 4, c2, k)

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


# Spatial pyramid pooling layer used in YOLOv3-SPP
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        self.cv1 = ConvBlock(c1, c1 // 2, 1, 1)
        self.cv2 = ConvBlock(c1 // 2 * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], 1)
        x = self.cv2(x)
        return x


class YoloV5(nn.Module):
    def __init__(self, w, h, c, num_class=90):
        super().__init__()
        assert w % 32 == 0 and h % 32 == 0, "image width and height must be a multiple of 32"
        self.num_class = num_class
        self.focus = Focus(c, 32, 3)
        self.conv1 = ConvBlock(32, 64, 3, 2)
        self.c3_1 = C3(64, 64)
        self.conv2 = ConvBlock(64, 128, 3, 2)
        self.c3_2 = C3(128, 128, 3)
        self.conv3 = ConvBlock(128, 256, 3, 2)
        self.c3_3 = C3(256, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3, 2)
        self.spp = SPP(512, 512)
        self.c3_4 = C3(512, 512, shortcut=False)
        self.conv5 = ConvBlock(512, 256)
        self.up1 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.c3_5 = C3(512, 256, shortcut=False)
        self.up2 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.conv6 = ConvBlock(256, 128)
        self.c3_6 = C3(256, 128, shortcut=False)
        self.conv7 = ConvBlock(128, 128, 3, 2)
        self.c3_7 = C3(256, 256, shortcut=False)
        self.conv8 = ConvBlock(256, 256, 3, 2)
        self.c3_8 = C3(512, 512, shortcut=False)
        self.conv17 = nn.Conv2d(128, (num_class + 4) * 3, 1)
        self.conv20 = nn.Conv2d(256, (num_class + 4) * 3, 1)
        self.conv23 = nn.Conv2d(512, (num_class + 4) * 3, 1)
        self.stride = torch.tensor([256 * x / w for x in [8, 16, 32]])
        self._initialize_detect_bias()

    def _initialize_detect_bias(self):
        for layer, stride in zip([self.conv17, self.conv20, self.conv23], self.stride):
            b = layer.bias.view(3, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / stride)**2)  # obj (8 objects per 640 image)
            b.data[:, 4:] += math.log(0.6 / (self.num_class - 0.99))  # cls
            layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        pdb.set_trace()
        x = self.c3_1(x)
        x = self.conv2(x)
        x_4 = self.c3_2(x)
        x = self.conv3(x_4)
        x_6 = self.c3_3(x)
        x = self.conv4(x_6)
        x = self.spp(x)
        x = self.c3_4(x)
        x_10 = self.conv5(x)
        x = self.up1(x_10)
        x = torch.cat([x, x_6], dim=1)
        x = self.c3_5(x)
        x_14 = self.conv6(x)
        x = self.up2(x_14)
        x = torch.cat([x, x_4], dim=1)
        x_17 = self.c3_6(x)
        x = self.conv7(x_17)
        x = torch.cat([x, x_14], dim=1)
        x_20 = self.c3_7(x)
        x = self.conv8(x_20)
        x = torch.cat([x, x_10], dim=1)
        x_23 = self.c3_8(x)
        out_17 = self.conv17(x_17)  # B, 255, h/8, w/8 - P3 stage
        out_20 = self.conv20(x_20)  # B, 255, h/16, w/16 - P4 stage
        out_23 = self.conv23(x_23)  # B, 255, h/32, w/32   - P5 stage
        # [B, c, h, w] -> [B, h, w, c], to make reshape meaningful on position
        out_17 = out_17.permute(0, 2, 3, 1)
        out_17 = out_17.reshape(out_17.size(0), -1, self.num_class + 4)
        out_20 = out_20.permute(0, 2, 3, 1)
        out_20 = out_20.reshape(out_20.size(0), -1, self.num_class + 4)
        out_23 = out_23.permute(0, 2, 3, 1)
        out_23 = out_23.reshape(out_23.size(0), -1, self.num_class + 4)
        results = torch.cat([out_17, out_20, out_23], dim=-2)
        return results


# This dataset selects 4 images and its bboxes
class PreMosaicDataset(Dataset):
    def __init__(self, mscoco_ds, image_size=640):
        self.mscoco_ds = mscoco_ds
        self.image_size = image_size
        self.op_ds = OpDataset(
            dataset=mscoco_ds,
            ops=[
                ReadImage(inputs="image", outputs="image"),
                LongestMaxSize(max_size=image_size,
                               image_in="image",
                               bbox_in="bbox",
                               bbox_params=BboxParams("coco", min_area=1.0))
            ],
            mode="train")

    def __len__(self):
        return len(self.mscoco_ds)

    def __getitem__(self, idx):
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        samples = [self.op_ds[i] for i in indices]
        images = [sample["image"] for sample in samples]
        bboxes = [sample["bbox"] for sample in samples]
        return {"image": images, "bbox": bboxes}


class PadCorners(NumpyOp):
    def forward(self, data, state):
        images, bboxes = data
        positions = ["topleft", "topright", "bottomleft", "bottomright"]
        images_new, bboxes_new = [], []
        for (pos, image, bbox) in zip(positions, images, bboxes):
            image_new, bbox_new = self._pad_image_corner(image, bbox, pos)
            images_new.append(image_new)
            bboxes_new.append(bbox_new)
        return images_new, bboxes_new

    def _pad_image_corner(self, image, bbox, pos):
        height, width = image.shape[0], image.shape[1]
        if height > width and (pos == "topleft" or pos == "bottomleft"):
            image, bbox = self._pad_left(image, bbox)
        elif height < width and (pos == "topleft" or pos == "topright"):
            image, bbox = self._pad_up(image, bbox)
        elif height > width and (pos == "topright" or pos == "bottomright"):
            image, bbox = self._pad_right(image, bbox)
        elif height < width and (pos == "bottomleft" or pos == "bottomright"):
            image, bbox = self._pad_down(image, bbox)
        return image, bbox

    def _pad_up(self, image, bbox, pad_value=114):
        pad_length = abs(image.shape[0] - image.shape[1])
        image = np.pad(image, [[pad_length, 0], [0, 0], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[1] = new_box[1] + pad_length
            bbox[i] = tuple(new_box)
        return image, bbox

    def _pad_down(self, image, bbox, pad_value=114):
        pad_length = abs(image.shape[0] - image.shape[1])
        image = np.pad(image, [[0, pad_length], [0, 0], [0, 0]], 'constant', constant_values=pad_value)
        return image, bbox

    def _pad_left(self, image, bbox, pad_value=114):
        pad_length = abs(image.shape[0] - image.shape[1])
        image = np.pad(image, [[0, 0], [pad_length, 0], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[0] = new_box[0] + pad_length
            bbox[i] = tuple(new_box)
        return image, bbox

    def _pad_right(self, image, bbox, pad_value=114):
        pad_length = abs(image.shape[0] - image.shape[1])
        image = np.pad(image, [[0, 0], [0, pad_length], [0, 0]], 'constant', constant_values=pad_value)
        return image, bbox


class CombineMosaic(NumpyOp):
    def forward(self, data, state):
        images, bboxes = data
        images_new = self._combine_images(images)
        bboxes_new = self._combine_boxes(bboxes, images)
        return images_new, bboxes_new

    def _combine_images(self, images):
        height, width, channel = images[0].shape
        images_new = np.full((2 * height, 2 * width, channel), fill_value=114, dtype=np.uint8)
        images_new[:height, :width] = images[0]  # top left
        images_new[:height, width:] = images[1]  # top right
        images_new[height:, :width] = images[2]  # bottom left
        images_new[height:, width:] = images[3]  # bottom right
        return images_new

    def _combine_boxes(self, bboxes, images):
        height, width, _ = images[0].shape
        bboxes_new = []
        for img_idx, bbox in enumerate(bboxes):
            for box in bbox:
                new_box = list(box)
                if img_idx == 1:  # top right
                    new_box[0] = new_box[0] + width
                elif img_idx == 2:  # bottom left
                    new_box[1] = new_box[1] + height
                elif img_idx == 3:  # bottom right
                    new_box[0] = new_box[0] + width
                    new_box[1] = new_box[1] + height
                bboxes_new.append(tuple(new_box))
        return bboxes_new


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


def get_yolo_anchor_box(width,
                        height,
                        P3_anchors=[(10, 13), (16, 30), (33, 23)],
                        P4_anchors=[(30, 61), (62, 45), (59, 119)],
                        P5_anchors=[(116, 90), (156, 198), (373, 326)]):
    """
    The anchors are the width, height in image space
    """
    assert height % 32 == 0 and width % 32 == 0
    shapes = [(height / 8, width / 8), (height / 16, width / 16), (height / 32, width / 32)]
    num_pixel = [int(np.prod(shape)) for shape in shapes]
    anchorbox = np.zeros((3 * np.sum(num_pixel), 4))
    anchor_idx = 0
    for shape, anchors in zip(shapes, [P3_anchors, P4_anchors, P5_anchors]):
        p_h, p_w = int(shape[0]), int(shape[1])
        base_y = 2**np.ceil(np.log2(height / p_h))
        base_x = 2**np.ceil(np.log2(width / p_w))
        for i in range(p_h):
            center_y = (i + 1 / 2) * base_y
            for j in range(p_w):
                center_x = (j + 1 / 2) * base_x
                for anchor in anchors:
                    anchorbox[anchor_idx, 0] = center_x - anchor[0] / 2  # x1
                    anchorbox[anchor_idx, 1] = center_y - anchor[1] / 2  # y1
                    anchorbox[anchor_idx, 2] = anchor[0]  # width
                    anchorbox[anchor_idx, 3] = anchor[1]  # height
                    anchor_idx += 1
    return np.float32(anchorbox), np.int32(num_pixel) * 3


class AnchorBox(NumpyOp):
    def __init__(self, width, height, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.anchorbox, _ = get_yolo_anchor_box(width, height)  # anchorbox is #num_anchor x 4

    def forward(self, data, state):
        target = self._generate_target(data)  # bbox is #obj x 5
        return np.float32(target)

    def _generate_target(self, bbox):
        object_boxes = bbox[:, :-1]  # num_obj x 4
        label = bbox[:, -1]  # num_obj x 1
        ious = self._get_iou(object_boxes, self.anchorbox)  # num_obj x num_anchor
        # now for each object in image, assign the anchor box with highest iou to them
        anchorbox_best_iou_idx = np.argmax(ious, axis=1)
        num_obj = ious.shape[0]
        for row in range(num_obj):
            ious[row, anchorbox_best_iou_idx[row]] = 0.99
        # next, begin the anchor box assignment based on iou
        anchor_to_obj_idx = np.argmax(ious, axis=0)  # num_anchor x 1
        anchor_best_iou = np.max(ious, axis=0)  # num_anchor x 1
        cls_gt = np.int32([label[idx] for idx in anchor_to_obj_idx])  # num_anchor x 1
        cls_gt[np.where(anchor_best_iou <= 0.4)] = -1  # background class
        cls_gt[np.where(np.logical_and(anchor_best_iou > 0.4, anchor_best_iou <= 0.5))] = -2  # ignore these examples
        # finally, calculate localization target
        single_loc_gt = object_boxes[anchor_to_obj_idx]  # num_anchor x 4
        gt_x1, gt_y1, gt_width, gt_height = np.split(single_loc_gt, 4, axis=1)
        ac_x1, ac_y1, ac_width, ac_height = np.split(self.anchorbox, 4, axis=1)
        dx1 = np.squeeze((gt_x1 - ac_x1) / ac_width)
        dy1 = np.squeeze((gt_y1 - ac_y1) / ac_height)
        dwidth = np.squeeze(np.log(gt_width / ac_width))
        dheight = np.squeeze(np.log(gt_height / ac_height))
        return np.array([dx1, dy1, dwidth, dheight, cls_gt]).T  # num_anchor x 5

    @staticmethod
    def _get_iou(boxes1, boxes2):
        """Computes the value of intersection over union (IoU) of two array of boxes.
        Args:
            box1 (array): first boxes in N x 4
            box2 (array): second box in M x 4
        Returns:
            float: IoU value in N x M
        """
        x11, y11, w1, h1 = np.split(boxes1, 4, axis=1)
        x21, y21, w2, h2 = np.split(boxes2, 4, axis=1)
        x12 = x11 + w1
        y12 = y11 + h1
        x22 = x21 + w2
        y22 = y21 + h2
        xmin = np.maximum(x11, np.transpose(x21))
        ymin = np.maximum(y11, np.transpose(y21))
        xmax = np.minimum(x12, np.transpose(x22))
        ymax = np.minimum(y12, np.transpose(y22))
        inter_area = np.maximum((xmax - xmin + 1), 0) * np.maximum((ymax - ymin + 1), 0)
        area1 = (w1 + 1) * (h1 + 1)
        area2 = (w2 + 1) * (h2 + 1)
        iou = inter_area / (area1 + area2.T - inter_area)
        return iou


class ShiftLabel(NumpyOp):
    def forward(self, data, state):
        # the label of COCO dataset starts from 1, shifting the start to 0
        bbox = np.array(data, dtype=np.float32)
        bbox[:, -1] = bbox[:, -1] - 1
        return bbox


class Rescale(NumpyOp):
    def forward(self, data, state):
        return np.float32(data / 255)


class YoloLoss(TensorOp):
    def forward(self, data, state):
        anchorbox, obj_pred = data
        loc_pred, cls_pred = obj_pred[..., :4], obj_pred[..., 4:]
        # anchorbox, cls_pred, loc_pred = data
        batch_size = anchorbox.size(0)
        focal_loss, l1_loss = 0.0, 0.0
        for idx in range(batch_size):
            single_loc_gt, single_cls_gt = anchorbox[idx][:, :-1], anchorbox[idx][:, -1].long()
            single_loc_pred, single_cls_pred = loc_pred[idx], cls_pred[idx]
            single_focal_loss, anchor_obj_bool = self.focal_loss(single_cls_gt, single_cls_pred)
            single_l1_loss = self.smooth_l1(single_loc_gt, single_loc_pred, anchor_obj_bool)
            focal_loss += single_focal_loss
            l1_loss += single_l1_loss
        focal_loss = focal_loss / batch_size
        l1_loss = l1_loss / batch_size
        total_loss = focal_loss + l1_loss
        return total_loss, focal_loss, l1_loss

    def focal_loss(self, single_cls_gt, single_cls_pred, alpha=0.25, gamma=2.0):
        # single_cls_gt shape: [num_anchor], single_cls_pred shape: [num_anchor, num_class]
        num_classes = single_cls_pred.size(-1)
        # gather the objects and background, discard the rest
        anchor_obj_bool = single_cls_gt >= 0
        anchor_background_obj_bool = single_cls_gt >= -1
        anchor_background_bool = single_cls_gt == -1
        # create one hot encoder, make -1 (background) and -2 (ignore) encoded as 0 in ground truth
        single_cls_gt[single_cls_gt < 0] = 0
        single_cls_gt = nn.functional.one_hot(single_cls_gt, num_classes=num_classes)
        single_cls_gt[anchor_background_bool] = 0
        single_cls_gt = single_cls_gt[anchor_background_obj_bool]  # remove all ignore cases
        single_cls_gt = single_cls_gt.view(-1)
        single_cls_pred = single_cls_pred[anchor_background_obj_bool]
        single_cls_pred = single_cls_pred.view(-1)
        single_cls_pred = torch.sigmoid(single_cls_pred)
        # compute the focal weight on each selected anchor box
        alpha_factor = torch.ones_like(single_cls_gt) * alpha
        alpha_factor = torch.where(single_cls_gt == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = torch.where(single_cls_gt == 1, 1 - single_cls_pred, single_cls_pred)
        focal_weight = alpha_factor * focal_weight**gamma / torch.sum(anchor_obj_bool)
        focal_loss = nn.functional.binary_cross_entropy(input=single_cls_pred,
                                                        target=single_cls_gt.float(),
                                                        weight=focal_weight.detach(),
                                                        reduction="sum")
        return focal_loss, anchor_obj_bool

    def smooth_l1(self, single_loc_gt, single_loc_pred, anchor_obj_bool, beta=0.1):
        # single_loc_gt shape: [num_anchor x 4], anchor_obj_idx shape:  [num_anchor x 4]
        single_loc_pred = single_loc_pred[anchor_obj_bool]  # anchor_obj_count x 4
        single_loc_gt = single_loc_gt[anchor_obj_bool]  # anchor_obj_count x 4
        single_loc_pred = single_loc_pred.view(-1)
        single_loc_gt = single_loc_gt.view(-1)
        loc_diff = torch.abs(single_loc_gt - single_loc_pred)
        loc_loss = torch.where(loc_diff < beta, 0.5 * loc_diff**2 / beta, loc_diff - 0.5 * beta)
        loc_loss = torch.sum(loc_loss) / torch.sum(anchor_obj_bool)
        return loc_loss


class PredictBox(TensorOp):
    """Convert network output to bounding boxes."""
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 mode=None,
                 input_shape=(512, 512, 3),
                 select_top_k=1000,
                 nms_max_outputs=100,
                 score_threshold=0.05):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.input_shape = input_shape
        self.select_top_k = select_top_k
        self.nms_max_outputs = nms_max_outputs
        self.score_threshold = score_threshold
        self.all_anchors, self.num_anchors_per_level = get_yolo_anchor_box(width=input_shape[1], height=input_shape[0])
        self.all_anchors = torch.Tensor(self.all_anchors)
        if torch.cuda.is_available():
            self.all_anchors = self.all_anchors.to("cuda")

    def forward(self, data, state):
        loc_pred, cls_pred = data[..., :4], data[..., 4:]
        batch_size = cls_pred.size(0)
        scores_pred, labels_pred = torch.max(cls_pred, dim=-1)
        # loc_pred -> loc_abs
        x1_abs = loc_pred[..., 0] * self.all_anchors[..., 2] + self.all_anchors[..., 0]
        y1_abs = loc_pred[..., 1] * self.all_anchors[..., 3] + self.all_anchors[..., 1]
        w_abs = torch.exp(loc_pred[..., 2]) * self.all_anchors[..., 2]
        h_abs = torch.exp(loc_pred[..., 3]) * self.all_anchors[..., 3]
        x2_abs, y2_abs = x1_abs + w_abs, y1_abs + h_abs
        # iterate over images
        final_results = []
        for idx in range(batch_size):
            scores_pred_single = scores_pred[idx]
            boxes_pred_single = torch.stack([x1_abs[idx], y1_abs[idx], x2_abs[idx], y2_abs[idx]], dim=-1)
            # iterate over each pyramid to select top 1000 anchor boxes
            start = 0
            top_idx = []
            for num_anchors_fpn_level in self.num_anchors_per_level:
                fpn_scores = scores_pred_single[start:start + num_anchors_fpn_level]
                _, selected_index = torch.topk(fpn_scores, min(self.select_top_k, int(num_anchors_fpn_level)))
                top_idx.append(selected_index + start)
                start += num_anchors_fpn_level
            top_idx = torch.cat([x.long() for x in top_idx])
            # perform nms
            nms_keep = torchvision.ops.nms(boxes_pred_single[top_idx], scores_pred_single[top_idx], iou_threshold=0.5)
            nms_keep = nms_keep[:self.nms_max_outputs]  # select the top nms outputs
            top_idx = top_idx[nms_keep]  # narrow the keep index
            results_single = [
                x1_abs[idx][top_idx],
                y1_abs[idx][top_idx],
                w_abs[idx][top_idx],
                h_abs[idx][top_idx],
                labels_pred[idx][top_idx].float(),
                scores_pred[idx][top_idx],
                torch.ones_like(x1_abs[idx][top_idx])
            ]
            # clip bounding boxes to image size
            results_single[0] = torch.clamp(results_single[0], min=0, max=self.input_shape[1])
            results_single[1] = torch.clamp(results_single[1], min=0, max=self.input_shape[0])
            results_single[2] = torch.clamp(results_single[2], min=0)
            results_single[2] = torch.where(results_single[2] > self.input_shape[1] - results_single[0],
                                            self.input_shape[1] - results_single[0],
                                            results_single[2])
            results_single[3] = torch.clamp(results_single[3], min=0)
            results_single[3] = torch.where(results_single[3] > self.input_shape[0] - results_single[1],
                                            self.input_shape[0] - results_single[1],
                                            results_single[3])
            # mark the select as 0 for any anchorbox with score lower than threshold
            results_single[-1] = torch.where(results_single[-2] > self.score_threshold,
                                             results_single[-1],
                                             torch.zeros_like(results_single[-1]))
            final_results.append(torch.stack(results_single, dim=-1))
        return torch.stack(final_results)


def lr_fn(step):
    if step < 2000:
        lr = (0.01 - 0.0002) / 2000 * step + 0.0002
    elif step < 120000:
        lr = 0.01
    elif step < 160000:
        lr = 0.001
    else:
        lr = 0.0001
    return lr / 2  # original batch_size 16, for 512 we have batch_size 8


def get_estimator(data_dir, model_dir=tempfile.mkdtemp(), batch_size=8, epochs=13):
    coco_train, coco_eval = mscoco.load_data(root_dir=data_dir)
    train_ds = PreMosaicDataset(coco_train, image_size=640)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=coco_eval,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image", mode="eval"),
            LongestMaxSize(max_size=1280,
                           image_in="image",
                           bbox_in="bbox",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="eval"),
            PadIfNeeded(min_height=1280,
                        min_width=1280,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="eval",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            PadCorners(inputs=("image", "bbox"), outputs=("image", "bbox"), mode="train"),
            CombineMosaic(inputs=("image", "bbox"), outputs=("image", "bbox"), mode="train"),
            PadIfNeeded(min_height=1920,
                        min_width=1920,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="train",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            RandomCrop(height=1280,
                       width=1280,
                       image_in="image",
                       bbox_in="bbox",
                       bbox_params=BboxParams("coco", min_area=1.0),
                       mode="train"),
            RandomScale(scale_limit=0.5,
                        image_in="image",
                        bbox_in="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="train"),
            Resize(height=1280,
                   width=1280,
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
            Rescale(inputs="image", outputs="image"),
            ShiftLabel(inputs="bbox", outputs="bbox"),
            AnchorBox(inputs="bbox", outputs="anchorbox", width=1280, height=1280),
            ChannelTranspose(inputs="image", outputs="image")
        ],
        pad_value=0)
    model = fe.build(model_fn=lambda: YoloV5(w=1280, h=1280, c=3, num_class=90),
                     optimizer_fn=lambda x: torch.optim.SGD(x, lr=1e-4, momentum=0.9, weight_decay=0.0001))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="image", outputs="obj_pred"),
        YoloLoss(inputs=("anchorbox", "obj_pred"), outputs=("total_loss", "focal_loss", "l1_loss")),
        UpdateOp(model=model, loss_name="total_loss"),
        PredictBox(input_shape=(1280, 1280, 3), inputs="obj_pred", outputs="pred", mode="eval")
    ])
    traces = [
        MeanAveragePrecision(num_classes=90, true_key='bbox', pred_key='pred', mode="eval"),
        BestModelSaver(model=model, save_dir=model_dir, metric='mAP', save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lr_fn)
    ]
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             traces=traces,
                             monitor_names=["l1_loss", "focal_loss"])
    return estimator

if __name__ == "__main__":
    model = YoloV5(w=1280, h=1280, c=3, num_class=80)
    inputs = torch.rand(1, 3, 1280, 1280)
    pred = model(inputs)
