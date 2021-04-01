import os
import pdb
import random

import cv2
import fastestimator as fe
import numpy as np
from albumentations import BboxParams
from fastestimator.op.numpyop import NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, RandomCrop, \
    RandomScale, Resize
from fastestimator.op.numpyop.univariate import ReadImage, ToArray
from torch.utils.data import Dataset


class CocoData(Dataset):
    def __init__(self, label_path):
        self.label_path = label_path
        self.files = os.listdir(label_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        gt_file = os.path.join(self.label_path, self.files[idx])
        image_file = gt_file.replace("/labels/", "/images/")
        image_file = image_file.replace(".txt", ".jpg")
        return {"image": image_file, "bbox": gt_file}


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


class Readtxt(NumpyOp):
    def forward(self, data, state):
        all_content = []
        with open(data, 'r') as f:
            for line in f:
                box = [float(x) for x in line.strip().split(' ')]  # class, x_c, y_c, w, h
                box.append(box.pop(0))  # x_c, y_c, w, h, class
                all_content.append(box)
        return all_content


class PadCorner(NumpyOp):
    def __init__(self, pos, inputs, outputs, mode):
        assert pos in {"topleft", "topright", "bottomleft", "bottomright"}
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.pos = pos

    def forward(self, data, state):
        image, bbox = data
        height, width = image.shape[0], image.shape[1]
        if height > width and (self.pos == "topleft" or self.pos == "bottomleft"):
            image, bbox = self._pad_left(image, bbox)
        elif height < width and (self.pos == "topleft" or self.pos == "topright"):
            image, bbox = self._pad_up(image, bbox)
        elif height > width and (self.pos == "topright" or self.pos == "bottomright"):
            image, bbox = self._pad_right(image, bbox)
        elif height < width and (self.pos == "bottomleft" or self.pos == "bottomright"):
            image, bbox = self._pad_down(image, bbox)
        return image, bbox

    def _pad_up(self, image, bbox, pad_value=114):
        height, width = image.shape[0], image.shape[1]
        pad_length = abs(height - width)
        image = np.pad(image, [[pad_length, 0], [0, 0], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[1] = (new_box[1] * height + pad_length) / (height + pad_length)
            new_box[3] = new_box[3] * height / (height + pad_length)
            bbox[i] = tuple(new_box)
        return image, bbox

    def _pad_down(self, image, bbox, pad_value=114):
        height, width = image.shape[0], image.shape[1]
        pad_length = abs(height - width)
        image = np.pad(image, [[0, pad_length], [0, 0], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[1] = (new_box[1] * height) / (height + pad_length)
            new_box[3] = new_box[3] * height / (height + pad_length)
            bbox[i] = tuple(new_box)
        return image, bbox

    def _pad_left(self, image, bbox, pad_value=114):
        height, width = image.shape[0], image.shape[1]
        pad_length = abs(height - width)
        image = np.pad(image, [[0, 0], [pad_length, 0], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[0] = (new_box[0] * width + pad_length) / (width + pad_length)
            new_box[2] = new_box[2] * width / (width + pad_length)
            bbox[i] = tuple(new_box)
        return image, bbox

    def _pad_right(self, image, bbox, pad_value=114):
        height, width = image.shape[0], image.shape[1]
        pad_length = abs(height - width)
        image = np.pad(image, [[0, 0], [0, pad_length], [0, 0]], 'constant', constant_values=pad_value)
        for i, box in enumerate(bbox):
            new_box = list(box)
            new_box[0] = (new_box[0] * width) / (width + pad_length)
            new_box[2] = new_box[2] * width / (width + pad_length)
            bbox[i] = tuple(new_box)
        return image, bbox


class CombineMosaic(NumpyOp):
    def forward(self, data, state):
        image1, image2, image3, image4, bbox1, bbox2, bbox3, bbox4 = data
        images = [image1, image2, image3, image4]
        bboxes = [bbox1, bbox2, bbox3, bbox4]
        images_new = self._combine_images(images)
        bboxes_new = self._combine_boxes(bboxes)
        return images_new, bboxes_new

    def _combine_images(self, images):
        height, width, channel = images[0].shape
        images_new = np.full((2 * height, 2 * width, channel), fill_value=114, dtype=np.uint8)
        images_new[:height, :width] = images[0]  # top left
        images_new[:height, width:] = images[1]  # top right
        images_new[height:, :width] = images[2]  # bottom left
        images_new[height:, width:] = images[3]  # bottom right
        return images_new

    def _combine_boxes(self, bboxes):
        bboxes_new = []
        for img_idx, bbox in enumerate(bboxes):
            for box in bbox:
                new_box = list(box)
                if img_idx == 0:  # top left
                    new_box[0] = new_box[0] / 2
                    new_box[1] = new_box[1] / 2
                elif img_idx == 1:  # top right
                    new_box[0] = new_box[0] / 2 + 0.5
                    new_box[1] = new_box[1] / 2
                elif img_idx == 2:  # bottom left
                    new_box[0] = new_box[0] / 2
                    new_box[1] = new_box[1] / 2 + 0.5
                elif img_idx == 3:  # bottom right
                    new_box[0] = new_box[0] / 2 + 0.5
                    new_box[1] = new_box[1] / 2 + 0.5
                new_box[2] = new_box[2] / 2
                new_box[3] = new_box[3] / 2
                bboxes_new.append(tuple(new_box))
        return bboxes_new


class Yolo2CocoBox(NumpyOp):
    def forward(self, data, state):
        image, yolo_bboxes = data
        width, height = image.shape[0], image.shape[1]
        new_boxes = []
        for xc, yc, w_ratio, h_ratio, classid in yolo_bboxes:
            w = int(w_ratio * width)
            h = int(h_ratio * height)
            x1 = int(xc * width - w / 2)
            y1 = int(yc * height - h / 2)
            if x1 + w > width or y1 + h > height:
                pdb.set_trace()
            new_boxes.append((x1, y1, w, h, classid))
        return new_boxes


class DebugOp(NumpyOp):
    def forward(self, data, state):
        pdb.set_trace()
        return data


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


def get_estimator():
    train_ds = CocoData("/data/data/public/MSCOCO2017/coco/labels/train2017")
    train_ds = PreMosaicDataset(mscoco_ds=train_ds)
    val_ds = CocoData("/data/data/public/MSCOCO2017/coco/labels/val2017")
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        batch_size=1,
        ops=[
            ReadImage(inputs="image1", outputs="image1", mode="train"),
            ReadImage(inputs="image2", outputs="image2", mode="train"),
            ReadImage(inputs="image3", outputs="image3", mode="train"),
            ReadImage(inputs="image4", outputs="image4", mode="train"),
            Readtxt(inputs="bbox1", outputs="bbox1", mode="train"),
            Readtxt(inputs="bbox2", outputs="bbox2", mode="train"),
            Readtxt(inputs="bbox3", outputs="bbox3", mode="train"),
            Readtxt(inputs="bbox4", outputs="bbox4", mode="train"),
            Yolo2CocoBox(inputs=("image1", "bbox1"), outputs="bbox1", mode="train"),
            Yolo2CocoBox(inputs=("image2", "bbox2"), outputs="bbox2", mode="train"),
            Yolo2CocoBox(inputs=("image3", "bbox3"), outputs="bbox3", mode="train"),
            Yolo2CocoBox(inputs=("image4", "bbox4"), outputs="bbox4", mode="train"),
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
                           bbox_params=BboxParams("coco"),
                           mode="train")
            # PadCorner(pos="topleft", inputs=("image1", "bbox1"), outputs=("image1", "bbox1"), mode="train"),
            # PadCorner(pos="topright", inputs=("image2", "bbox2"), outputs=("image2", "bbox2"), mode="train"),
            # PadCorner(pos="bottomleft", inputs=("image3", "bbox3"), outputs=("image3", "bbox3"), mode="train"),
            # PadCorner(pos="bottomright", inputs=("image4", "bbox4"), outputs=("image4", "bbox4"), mode="train"),
            # CombineMosaic(inputs=("image1", "image2", "image3", "image4", "bbox1", "bbox2", "bbox3", "bbox4"),
            #               outputs=("image", "bbox"),
            #               mode="train")
            # PadIfNeeded(min_height=1920,
            #             min_width=1920,
            #             image_in="image",
            #             bbox_in="bbox",
            #             bbox_params=BboxParams("yolo", min_area=1.0),
            #             mode="train",
            #             border_mode=cv2.BORDER_CONSTANT,
            #             value=(114, 114, 114)),
            # RandomCrop(height=1280,
            #            width=1280,
            #            image_in="image",
            #            bbox_in="bbox",
            #            bbox_params=BboxParams("yolo", min_area=1.0),
            #            mode="train"),
            # RandomScale(scale_limit=0.5,
            #             image_in="image",
            #             bbox_in="bbox",
            #             bbox_params=BboxParams("yolo", min_area=1.0),
            #             mode="train"),
            # Resize(height=640,
            #        width=640,
            #        image_in="image",
            #        bbox_in="bbox",
            #        bbox_params=BboxParams("yolo", min_area=1.0),
            #        mode="train"),
            # Sometimes(
            #     HorizontalFlip(image_in="image",
            #                    bbox_in="bbox",
            #                    bbox_params=BboxParams("yolo", min_area=1.0),
            #                    mode="train")),
            # HSVAugment(inputs="image", outputs="image", mode="train"),
            # Rescale(inputs="image", outputs="image"),
            # ToArray(inputs="bbox", outputs="bbox", dtype="float32"),
        ],
        num_process=0)
    data = pipeline.get_results(mode="train")
    pdb.set_trace()


get_estimator()
