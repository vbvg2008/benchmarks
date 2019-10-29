import json
import os

import numpy as np

import fastestimator as fe
from fastestimator.architecture.retinanet import RetinaNet, get_fpn_anchor_box, get_target
from fastestimator.dataset.mscoco import load_data
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop import ImageReader, ResizeImageAndBbox


class String2List(NumpyOp):
    # this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data, state):
        for idx, elem in enumerate(data):
            data[idx] = np.array([float(x) for x in elem[1:-1].split(',')])
        return data


class GenerateTarget(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.anchorbox = get_fpn_anchor_box(input_shape=(512, 512, 3))

    def forward(self, data, state):
        obj_label, x1, y1, width, height = data
        target_cls, target_loc = get_target(self.anchorbox, obj_label, x1, y1, width, height)
        return target_cls, target_loc


class ConvertToInt32(NumpyOp):
    # this thing converts '[1, 2, 3]' into np.array([1, 2, 3])
    def forward(self, data, state):
        for idx, elem in enumerate(data):
            data[idx] = np.int32(elem)
        return data


# sample_data = {
#     "image": ["val2017/000000089045.jpg"],
#     "num_obj": [9],
#     "x1": ["[432.5, 65.81, 191.94, 594.64, 446.79, 591.69, 214.54, 513.71, 234.46]"],
#     "y1": ["[122.93, 188.28, 132.63, 215.24, 157.56, 122.53, 54.63, 148.13, 216.31]"],
#     "width": ["[42.04, 57.58, 277.84, 43.77, 20.84, 19.87, 16.11, 126.29, 87.82]"],
#     "height": ["[45.71, 181.88, 96.88, 61.1, 10.97, 30.91, 22.59, 18.43, 32.34]"],
#     "obj_label": ["[64, 62, 63, 63, 86, 62, 85, 67, 84]"],
#     "obj_mask": ["mask_val2017/000000089045.png"]
# }

_, val_csv, path = load_data("/data/data")

writer = fe.RecordWriter(
    save_dir="/data/data/coco_tfrecord",
    train_data=val_csv,
    ops=[
        ImageReader(inputs="image", parent_path=path, outputs="image"),
        String2List(inputs=["x1", "y1", "width", "height", "obj_label"],
                    outputs=["x1", "y1", "width", "height", "obj_label"]),
        ResizeImageAndBbox(target_size=(512, 512),
                           keep_ratio=True,
                           inputs=["image", "x1", "y1", "width", "height"],
                           outputs=["image", "x1", "y1", "width", "height"]),
        GenerateTarget(inputs=("obj_label", "x1", "y1", "width", "height"), outputs=("target_cls", "target_loc")),
        ConvertToInt32(inputs=["num_obj", "x1", "y1", "width", "height", "obj_label", "target_cls"],
                       outputs=["num_obj", "x1", "y1", "width", "height", "obj_label", "target_cls"])
    ],
    compression="GZIP",
    write_feature=["image", "x1", "y1", "width", "height", "obj_label", "target_cls", "target_loc", "num_obj"])

writer.write()
# results = writer.transform(data=sample_data, mode="train")

# for key, value in results.items():
#     value = np.array(value[0])
#     print(key)
#     print(value.dtype)

# np.savez("/data/testdata", **results)
