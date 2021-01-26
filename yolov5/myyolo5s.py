import math
import pdb
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations import BboxParams
from fastestimator.dataset.data import mscoco
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.op.numpyop.multivariate import LongestMaxSize
from fastestimator.op.numpyop.univariate import ReadImage
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


class MosaicDataset(Dataset):
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
        image_mosaic, bbox_mosaic = self._create_mosaic(images, bboxes)
        return {"image": image_mosaic, "bbox": bbox_mosaic}

    def _create_mosaic(self, images, bboxes):
        yc = int(random.uniform(self.image_size // 2, 2 * self.image_size - self.image_size // 2))
        xc = int(random.uniform(self.image_size // 2, 2 * self.image_size - self.image_size // 2))
        image_mosaic = np.full((self.image_size * 2, self.image_size * 2, 3), fill_value=114, dtype=np.uint8)
        for i, image in enumerate(images):
            h, w = image.shape[0], image.shape[1]
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.image_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.image_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.image_size * 2), min(self.image_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            image_mosaic[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pdb.set_trace()
        return image_mosaic, bboxes


if __name__ == "__main__":
    # model = YoloV5(input_shape=(256, 256, 3))

    # inputs = torch.rand(1, 3, 256, 256)
    # pred = model(inputs)
    # pdb.set_trace()
    coco_train, coco_eval = mscoco.load_data(root_dir="/data/data/public")
    train_ds = MosaicDataset(coco_train, image_size=640)
    train_ds[2]
