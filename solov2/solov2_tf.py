"""
pip install paddlepaddle
clone paddledetection: git clone https://github.com/PaddlePaddle/PaddleDetection.git
pip install -r requirements.txt
cd PaddleDetection
python setup.py install
pip install
python ppdet/modeling/tests/test_architectures.py
python tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml
"""
import pdb

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.ndimage.measurements import center_of_mass
from tensorflow.keras import layers

import fastestimator as fe
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, LongestMaxSize, PadIfNeeded, Resize
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop import Average
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.tensorop import LambdaOp, TensorOp


def fpn(C2, C3, C4, C5):
    # lateral conv
    P5 = layers.Conv2D(256, kernel_size=1)(C5)
    P5_up = layers.UpSampling2D()(P5)
    P4 = layers.Conv2D(256, kernel_size=1)(C4)
    P4 = P4 + P5_up
    P4_up = layers.UpSampling2D()(P4)
    P3 = layers.Conv2D(256, kernel_size=1)(C3)
    P3 = P3 + P4_up
    P3_up = layers.UpSampling2D()(P3)
    P2 = layers.Conv2D(256, kernel_size=1)(C2)
    P2 = P2 + P3_up
    # fpn conv
    P5 = layers.Conv2D(256, kernel_size=3, padding="same")(P5)
    P4 = layers.Conv2D(256, kernel_size=3, padding="same")(P4)
    P3 = layers.Conv2D(256, kernel_size=3, padding="same")(P3)
    P2 = layers.Conv2D(256, kernel_size=3, padding="same")(P2)
    return P2, P3, P4, P5


def pad_with_coord(data):
    data_shape = tf.shape(data)
    batch_size, height, width = data_shape[0], data_shape[1], data_shape[2]
    x = tf.cast(tf.linspace(-1, 1, num=width), data.dtype)
    x = tf.tile(x[tf.newaxis, tf.newaxis, ..., tf.newaxis], [batch_size, height, 1, 1])
    y = tf.cast(tf.linspace(-1, 1, num=height), data.dtype)
    y = tf.tile(y[tf.newaxis, ..., tf.newaxis, tf.newaxis], [batch_size, 1, width, 1])
    data = tf.concat([data, x, y], axis=-1)
    return data


def conv_norm(x, filters, kernel_size=3, groups=32):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = tfa.layers.GroupNormalization(groups=groups, epsilon=1e-5)(x)
    return x


def solov2_head_model(stacked_convs=4, ch_in=258, ch_feature=512, ch_kernel_out=256, num_classes=80):
    inputs = layers.Input(shape=(None, None, ch_in))
    feature_kernel = inputs
    feature_cls = inputs[..., :-2]
    for _ in range(stacked_convs):
        feature_kernel = tf.nn.relu(conv_norm(feature_kernel, filters=ch_feature))
        feature_cls = tf.nn.relu(conv_norm(feature_cls, filters=ch_feature))
    feature_kernel = layers.Conv2D(filters=ch_kernel_out,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01))(feature_kernel)
    feature_cls = layers.Conv2D(filters=num_classes,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.initializers.constant(np.log(1 / 99)))(feature_cls)
    return tf.keras.Model(inputs=inputs, outputs=[feature_kernel, feature_cls])


def solov2_head(P2, P3, P4, P5, num_classes=80):
    head_model = solov2_head_model(num_classes=num_classes)
    # applying maxpool first for P2
    P2 = layers.MaxPool2D()(P2)
    features = [P2, P3, P4, P5, P5]
    grid_sizes = [40, 36, 24, 16, 12]
    feat_kernel_list, feat_cls_list = [], []
    for feature, grid_size in zip(features, grid_sizes):
        feature = pad_with_coord(feature)
        feature = tf.image.resize(feature, size=(grid_size, grid_size))
        feat_kernel, feat_cls = head_model(feature)
        feat_kernel_list.append(feat_kernel)
        feat_cls_list.append(feat_cls)
    return feat_cls_list, feat_kernel_list


def solov2_maskhead(P2, P3, P4, P5, mid_ch=128, out_ch=256):
    # first level
    P2 = tf.nn.relu(conv_norm(P2, filters=mid_ch))
    # second level
    P3 = tf.nn.relu(conv_norm(P3, filters=mid_ch))
    P3 = layers.UpSampling2D()(P3)
    # third level
    P4 = tf.nn.relu(conv_norm(P4, filters=mid_ch))
    P4 = layers.UpSampling2D()(P4)
    P4 = tf.nn.relu(conv_norm(P4, filters=mid_ch))
    P4 = layers.UpSampling2D()(P4)
    # top level, add coordinate
    P5 = tf.nn.relu(conv_norm(pad_with_coord(P5), filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    P5 = tf.nn.relu(conv_norm(P5, filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    P5 = tf.nn.relu(conv_norm(P5, filters=mid_ch))
    P5 = layers.UpSampling2D()(P5)
    seg_outputs = tf.nn.relu(conv_norm(P2 + P3 + P4 + P5, filters=out_ch, kernel_size=1))
    return seg_outputs


def solov2(input_shape=(None, None, 3), num_classes=80):
    inputs = tf.keras.Input(shape=input_shape)
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs, pooling=None)
    assert resnet50.layers[38].name == "conv2_block3_out"
    C2 = resnet50.layers[38].output
    assert resnet50.layers[80].name == "conv3_block4_out"
    C3 = resnet50.layers[80].output
    assert resnet50.layers[142].name == "conv4_block6_out"
    C4 = resnet50.layers[142].output
    assert resnet50.layers[-1].name == "conv5_block3_out"
    C5 = resnet50.layers[-1].output
    P2, P3, P4, P5 = fpn(C2, C3, C4, C5)
    feat_seg = solov2_maskhead(P2, P3, P4, P5)  # [B, h/4, w/4, 256]
    feat_cls_list, feat_kernel_list = solov2_head(P2, P3, P4, P5, num_classes=num_classes)  # [B, grid, grid, 80], [B, grid, grid, 256]
    model = tf.keras.Model(inputs=inputs, outputs=[feat_seg, feat_cls_list, feat_kernel_list])
    return model


class MergeMask(NumpyOp):
    def forward(self, data, state):
        data = np.stack(data, axis=-1)
        return data


class CategoryID2ClassID(NumpyOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(category, list(range(80)))}

    def forward(self, data, state):
        classes = np.array([self.mapping[x[-1]] for x in data], dtype=np.int32)
        return classes


class Gt2Target(NumpyOp):
    def __init__(self,
                 inputs,
                 outputs,
                 mode=None,
                 im_size=1024,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
                 coord_sigma=0.05,
                 sampling_ratio=4.0):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.im_size = im_size
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio
        missing_category = [66, 68, 69, 71, 12, 45, 83, 26, 29, 30]
        category = [x for x in range(1, 91) if not x in missing_category]
        self.mapping = {k: v for k, v in zip(category, list(range(80)))}
        self.mask_size = int(im_size / sampling_ratio)

    def forward(self, data, state):
        masks, bboxes = data
        bboxes = np.array(bboxes, dtype="float32")
        masks = np.transpose(masks, [2, 0, 1])  # (H, W, #objects) -> (#objects, H, W)
        masks, bboxes = self.remove_empty_gt(masks, bboxes)
        # 91 classes -> 80 classes that starts from 1
        classes = np.array([self.mapping[int(x[-1])] + 1 for x in bboxes], dtype=np.int32)
        widths, heights = bboxes[:, 2], bboxes[:, 3]
        gt_match = []  # number of objects x (grid_idx, height_idx, width_idx, exist)
        for width, height, mask in zip(widths, heights, masks):
            object_match = []
            object_scale = np.sqrt(width * height)
            center_h, center_w = center_of_mass(mask)
            for grid_idx, ((lower_scale, upper_scale), num_grid) in enumerate(zip(self.scale_ranges, self.num_grids)):
                grid_matched = (object_scale >= lower_scale) & (object_scale <= upper_scale)
                if grid_matched:
                    w_delta, h_delta = 0.5 * width * self.coord_sigma, 0.5 * height * self.coord_sigma
                    coord_h, coord_w = int(center_h / mask.shape[0] * num_grid), int(center_w / mask.shape[1] * num_grid)
                    # each object will have some additional area of effect
                    top_box_extend = max(0, int((center_h - h_delta) / mask.shape[0] * num_grid))
                    down_box_extend = min(num_grid - 1, int((center_h + h_delta) / mask.shape[0] * num_grid))
                    left_box_extend = max(0, int((center_w - w_delta) / mask.shape[1] * num_grid))
                    right_box_extend = min(num_grid - 1, int((center_w + w_delta) / mask.shape[0] * num_grid))
                    # make sure the additional area of effect is at most 1 grid more
                    top_box_extend = max(top_box_extend, coord_h - 1)
                    down_box_extend = min(down_box_extend, coord_h + 1)
                    left_box_extend = max(left_box_extend, coord_w - 1)
                    right_box_extend = min(right_box_extend, coord_w + 1)
                    object_match.extend([(grid_idx, y, x, 1) for y in range(top_box_extend, down_box_extend + 1)
                                         for x in range(left_box_extend, right_box_extend + 1)])
            gt_match.append(object_match)
        gt_match = self.pad_match(gt_match)  #num_object x num_matches x [grid_idx, heihght_idx, width_idx, exist]
        return gt_match, masks, classes

    def pad_match(self, gt_match):
        max_num_matches = max([len(match) for match in gt_match])
        for match in gt_match:
            match.extend([(0, 0, 0, 0) for _ in range(max_num_matches - len(match))])
        return np.array(gt_match, dtype="int32")

    def remove_empty_gt(self, masks, bboxes):
        num_objects = masks.shape[0]
        non_empty_mask = np.sum(masks.reshape(num_objects, -1), axis=1) > 0
        return masks[non_empty_mask], bboxes[non_empty_mask]


class Normalize(TensorOp):
    def __init__(self, inputs, outputs, mean, std, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.mean = tf.convert_to_tensor(mean)
        self.std = tf.convert_to_tensor(std)

    def forward(self, data, state):
        data = (data / 255 - self.mean) / self.std
        return data


class Solov2Loss(TensorOp):
    def __init__(self, level, grid_dim, inputs, outputs, mode=None, num_class=80):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.level = level
        self.grid_dim = grid_dim
        self.num_class = num_class

    def forward(self, data, state):
        masks, classes, gt_match, feat_segs, feat_clss, kernels = data
        cls_loss, grid_object_maps = tf.map_fn(fn=lambda x: self.get_cls_loss(x[0], x[1], x[2]),
                             elems=(classes, feat_clss, gt_match),
                             fn_output_signature=(tf.float32, tf.float32))
        # seg_loss = tf.map_fn(fn=lambda x: self.get_seg_loss(x[0], x[1], x[2], x[3], x[4]),
        #                      elems=(classes, masks, feat_segs, kernels, gt_match),
        #                      fn_output_signature=tf.float32)
        seg_loss = []
        batch_size = classes.shape[0]
        for idx in range(batch_size):
            seg_loss.append(self.get_seg_loss(masks[idx], feat_segs[idx], kernels[idx], grid_object_maps[idx]))
        return tf.reduce_mean(cls_loss)

    def get_seg_loss(self, mask, feat_seg, kernel, grid_object_map):
        pdb.set_trace()
        indices = tf.where(grid_object_map[..., 0] > 0)
        object_indices = tf.cast(tf.gather_nd(grid_object_map, indices)[:, 1], tf.int32)
        mask_gt = tf.gather(mask, object_indices)
        active_kernel = tf.gather_nd(kernel, indices)
        mask_pred = tf.matmul(active_kernel, feat_seg)
        coord_label = tf.concat(
            [tf.gather_nd(match, indices)[:, 1:3], tf.gather(cls_gt, indices[:, 0])[..., tf.newaxis]], axis=-1)

    def get_cls_loss(self, cls_gt, feat_cls, match):
        cls_gt = tf.cast(cls_gt, feat_cls.dtype)
        match, cls_gt = match[cls_gt > 0], cls_gt[cls_gt > 0]  # remove the padded object
        feat_cls_gts_raw = tf.map_fn(fn=lambda x: self.assign_cls_feat(x[0], x[1]),
                                     elems=(match, cls_gt),
                                     fn_output_signature=tf.float32)
        # TODO: if there are multiple objects overlapping on same grid point, randomly choose one
        # reduce the gt for all objects into single grid
        feat_cls_gts = tf.reduce_max(feat_cls_gts_raw, axis=0)
        object_idx = tf.cast(tf.math.argmax(feat_cls_gts_raw, axis=0), feat_cls_gts.dtype)
        grid_object_map = tf.stack([feat_cls_gts, object_idx], axis=-1)
        # classification loss
        feat_cls_gts = tf.one_hot(tf.cast(feat_cls_gts, tf.int32), depth=self.num_class + 1)[..., 1:]
        loss = tf.losses.BinaryCrossentropy(reduction='sum')(feat_cls_gts, tf.sigmoid(feat_cls))
        return loss, grid_object_map

    def assign_cls_feat(self, grid_match_info, cls_gt_obj):
        match_bool = tf.logical_and(tf.reduce_sum(grid_match_info, axis=-1) > 0, grid_match_info[:, 0] == self.level)
        grid_match_info = grid_match_info[match_bool]
        grid_indices = grid_match_info[:, 1:3]
        num_indices = tf.shape(grid_indices)[0]
        feat_cls_gt = tf.scatter_nd(grid_indices, tf.fill([num_indices], cls_gt_obj), (self.grid_dim, self.grid_dim))
        return feat_cls_gt


def get_estimator(data_dir):
    train_ds, val_ds = mscoco.load_data(root_dir=data_dir, load_masks=True)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=val_ds,
        batch_size=4,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            MergeMask(inputs="mask", outputs="mask"),
            LongestMaxSize(max_size=1024, image_in="image", mask_in="mask", bbox_in="bbox", bbox_params="coco"),
            PadIfNeeded(min_height=1024,
                        min_width=1024,
                        image_in="image",
                        mask_in="mask",
                        bbox_in="bbox",
                        bbox_params="coco",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
            Sometimes(HorizontalFlip(image_in="image", mask_in="mask", bbox_in="bbox", bbox_params="coco",
                                     mode="train")),
            Resize(height=256, width=256, image_in='mask'),  # downscale mask by 1/4 for memory efficiency
            Gt2Target(inputs=("mask", "bbox"), outputs=("gt_match", "mask", "classes")),
            Delete(keys=("bbox", "image_id"))
        ],
        pad_value=0)
    model = fe.build(model_fn=lambda: solov2(input_shape=(1024, 1024, 3)), optimizer_fn="adam")
    network = fe.Network(ops=[
        Normalize(inputs="image", outputs="image", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ModelOp(model=model, inputs="image", outputs=("feat_seg", "feat_cls_list", "feat_kernel_list")),
        LambdaOp(fn=lambda x: x, inputs="feat_cls_list", outputs=("cls1", "cls2", "cls3", "cls4", "cls5")),
        LambdaOp(fn=lambda x: x, inputs="feat_kernel_list", outputs=("k1", "k2", "k3", "k4", "k5")),
        Solov2Loss(0, 40, inputs=("mask", "classes", "gt_match", "feat_seg", "cls1", "k1"), outputs="loss1"),
        Solov2Loss(1, 36, inputs=("mask", "classes", "gt_match", "feat_seg", "cls2", "k2"), outputs="loss2"),
        Solov2Loss(2, 24, inputs=("mask", "classes", "gt_match", "feat_seg", "cls3", "k3"), outputs="loss3"),
        Solov2Loss(3, 16, inputs=("mask", "classes", "gt_match", "feat_seg", "cls4", "k4"), outputs="loss4"),
        Solov2Loss(4, 12, inputs=("mask", "classes", "gt_match", "feat_seg", "cls5", "k5"), outputs="loss5"),
        Average(inputs=("loss1", "loss2", "loss3", "loss4", "loss5"), outputs="total_loss"),
        UpdateOp(model=model, loss_name="total_loss")
    ])
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=1)
    return estimator
