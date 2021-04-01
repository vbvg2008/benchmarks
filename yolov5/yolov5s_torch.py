import math
import pdb
import random

import cv2
import fastestimator as fe
import numpy as np
import torch
import torch.nn as nn
from albumentations import BboxParams
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import CenterCrop, HorizontalFlip, LongestMaxSize
from fastestimator.op.numpyop.univariate import ChannelTranspose, ReadImage, ToArray
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
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


class Rescale(NumpyOp):
    def forward(self, data, state):
        return np.float32(data / 255)


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


class Coco2Yolo(NumpyOp):
    def forward(self, data, state):
        x1, y1, width, height, label = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        xc = (x1 + width / 2) / 640
        yc = (y1 + height / 2) / 640
        width_ratio = width / 640
        height_ratio = height / 640
        box_yolo = np.stack((label, xc, yc, width_ratio, height_ratio), axis=1)
        return box_yolo


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


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YoloV5(nn.Module):
    def __init__(self, w, h, c, num_class=80):
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
        self.conv17 = nn.Conv2d(128, (num_class + 5) * 3, 1)
        self.conv20 = nn.Conv2d(256, (num_class + 5) * 3, 1)
        self.conv23 = nn.Conv2d(512, (num_class + 5) * 3, 1)
        self.stride = torch.tensor([8, 16, 32])
        self._initialize_detect_bias()

    def _initialize_detect_bias(self):
        for layer, stride in zip([self.conv17, self.conv20, self.conv23], self.stride):
            b = layer.bias.view(3, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / stride)**2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_class - 0.99))  # cls
            layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
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
        out = [out_17, out_20, out_23]
        for i, x in enumerate(out):
            bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out[i] = x.view(bs, 3, self.num_class + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return out


class ComputeLoss(TensorOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_anchor = 3
        self.num_layer_out = 3
        self.anchor_t = 4.0
        self.gr = 1.0
        self.stride = torch.tensor([8, 16, 32])
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)
        self.balance = [4.0, 1.0, 0.4]
        self.box_multi, self.obj_multi, self.cls_multi = 0.05, 1.0, 0.5
        self.cls_pw, self.obj_pw = 1.0, 1.0

    def build(self, framework, device):
        self.anchors = torch.tensor([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                     [116, 90, 156, 198, 373, 326]]).float().view(self.num_layer_out, -1, 2)
        self.anchors /= self.stride.view(-1, 1, 1)
        self.anchors = self.anchors.to(device)
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cls_pw], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.obj_pw], device=device))

    def forward(self, data, state):
        y_pred, targets = data
        targets = self._preprocess_targets(targets)
        tcls, tbox, indices, anchors = self.prepare_targets(y_pred, targets)
        lcls = torch.zeros((), device=targets.device)
        lbox = torch.zeros((), device=targets.device)
        lobj = torch.zeros((), device=targets.device)
        # Losses
        for i, pi in enumerate(y_pred):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=targets.device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # Classification
                t = torch.full_like(ps[:, 5:], self.cn, device=targets.device)  # targets
                t[range(n), tcls[i]] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t)  # BCE
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        loss = lbox * self.box_multi + lobj * self.obj_multi + lcls * self.cls_multi
        return loss * tobj.shape[0]

    def _preprocess_targets(self, targets):
        targets_unpad = [target[torch.sum(target, axis=1) > 0] for target in targets]
        targets_with_idx = [
            torch.cat([idx * torch.ones(target.size(0), 1, dtype=targets.dtype, device=targets.device), target], dim=1)
            for idx,
            target in enumerate(targets_unpad)
        ]
        targets_new = torch.cat(targets_with_idx, dim=0)
        return targets_new

    def prepare_targets(self, pred, targets):
        #input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  #targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(self.num_anchor,
                          device=targets.device).float().view(self.num_anchor,
                                                              1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(self.num_anchor, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]  # j,k,l,m
            ],
            device=targets.device).float() * g  # offsets
        for i in range(self.num_layer_out):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU


def get_estimator(data_dir="/data/data/public/COCO2017/"):
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir)
    train_ds = PreMosaicDataset(mscoco_ds=train_ds)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        # eval_data=val_ds,
        batch_size=40,
        ops=[
            ReadImage(inputs=("image1", "image2", "image3", "image4"),
                      outputs=("image1", "image2", "image3", "image4"),
                      mode="train"),
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
            Rescale(inputs="image", outputs="image"),
            ToArray(inputs="bbox", outputs="bbox", dtype="float32"),
            CategoryID2ClassID(inputs="bbox", outputs="bbox"),
            Coco2Yolo(inputs="bbox", outputs="bbox"),
            ChannelTranspose(inputs="image", outputs="image"),
            Delete(keys=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4"))
        ],
        pad_value=0)
    model = fe.build(lambda: YoloV5(w=640, h=640, c=3), optimizer_fn="sgd")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="image", outputs="y_pred"),
        ComputeLoss(inputs=("y_pred", "bbox"), outputs="loss"),
        UpdateOp(model=model, loss_name="loss")
    ])
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=2)
    return estimator


if __name__ == "__main__":
    # model = YoloV5(w=256, h=256, c=3)
    # inputs = torch.rand(1, 3, 256, 256)
    # pred = model(inputs)
    # pdb.set_trace()
    get_estimator()
