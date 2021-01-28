import json
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
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, RandomCrop, \
    RandomScale, Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, ReadImage, ToArray
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
    def __init__(self, input_shape=(256, 256, 3), num_class=80):
        super().__init__()
        w, h, c = input_shape
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
        self.stride = torch.tensor([256 * x / w for x in [8, 16, 32]])
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
        out_17 = self.conv17(x_17)  # B, 255, 32, 32
        out_20 = self.conv20(x_20)  # B, 255, 16, 16
        out_23 = self.conv23(x_23)  # B, 255, 8, 8
        # [B, 255, h, w] -> [B, 3, h, w, 85]
        out_17 = out_17.view(-1, 3, self.num_class + 5, out_17.shape[-2], out_17.shape[-1])
        out_17 = out_17.permute(0, 1, 3, 4, 2).contiguous()
        out_20 = out_20.view(-1, 3, self.num_class + 5, out_20.shape[-2], out_20.shape[-1])
        out_20 = out_20.permute(0, 1, 3, 4, 2).contiguous()
        out_23 = out_23.view(-1, 3, self.num_class + 5, out_23.shape[-2], out_23.shape[-1])
        out_23 = out_23.permute(0, 1, 3, 4, 2).contiguous()
        return [out_17, out_20, out_23]


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
                               image_out="image",
                               bbox_in="bbox",
                               bbox_out="bbox",
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


class PadCorners(fe.op.numpyop.NumpyOp):
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


class CombineMosaic(fe.op.numpyop.NumpyOp):
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


class HSVAugment(fe.op.numpyop.NumpyOp):
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


class DebugOp(fe.op.numpyop.NumpyOp):
    def forward(self, data, stte):
        image = data
        pdb.set_trace()
        return image


def get_estimator(image_size=640, batch_size=4):
    with open("class.json", 'r') as f:
        class_map = json.load(f)
    coco_train, coco_eval = mscoco.load_data(root_dir="/data/data/public")
    train_ds = PreMosaicDataset(coco_train, image_size=image_size)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=coco_eval,
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", outputs="image", mode="eval"),
            LongestMaxSize(max_size=image_size,
                           image_in="image",
                           image_out="image",
                           bbox_in="bbox",
                           bbox_out="bbox",
                           bbox_params=BboxParams("coco", min_area=1.0),
                           mode="eval"),
            PadCorners(inputs=("image", "bbox"), outputs=("image", "bbox"), mode="train"),
            CombineMosaic(inputs=("image", "bbox"), outputs=("image", "bbox"), mode="train"),
            PadIfNeeded(min_height=1920,
                        min_width=1920,
                        image_in="image",
                        image_out="image",
                        bbox_in="bbox",
                        bbox_out="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="train",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)),
            RandomCrop(height=1280,
                       width=1280,
                       image_in="image",
                       image_out="image",
                       bbox_in="bbox",
                       bbox_out="bbox",
                       bbox_params=BboxParams("coco", min_area=1.0),
                       mode="train"),
            RandomScale(scale_limit=0.5,
                        image_in="image",
                        image_out="image",
                        bbox_in="bbox",
                        bbox_out="bbox",
                        bbox_params=BboxParams("coco", min_area=1.0),
                        mode="train"),
            Resize(height=image_size * 2,
                   width=image_size * 2,
                   image_in="image",
                   image_out="image",
                   bbox_in="bbox",
                   bbox_out="bbox",
                   bbox_params=BboxParams("coco", min_area=1.0),
                   mode="train"),
            Sometimes(
                HorizontalFlip(image_in="image",
                               image_out="image",
                               bbox_in="bbox",
                               bbox_out="bbox",
                               bbox_params=BboxParams("coco", min_area=1.0),
                               mode="train")),
            HSVAugment(inputs="image", outputs="image", mode="train"),
            ToArray(inputs="bbox", outputs="bbox")
        ],
        pad_value=0)
    data = pipeline.get_results(mode="train")
    pdb.set_trace()
    img = data["image"].numpy()
    bbox = data["bbox"].numpy()
    new_box = []
    for box in bbox[0]:
        new_box.append(list(box[:4]) + [class_map[str(int(box[4]))]])
    new_box = np.array([new_box])
    img_visualize = fe.util.ImgData(x=[img, new_box])
    ig = img_visualize.paint_figure(save_path="object_results_5.png")


if __name__ == "__main__":
    get_estimator()
