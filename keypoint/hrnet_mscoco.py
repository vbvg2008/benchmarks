import pdb

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage.measurements import center_of_mass

import fastestimator as fe
from fastestimator.architecture.tensorflow import UNet
from fastestimator.dataset import NumpyDataset
from fastestimator.dataset.data.mscoco import load_data
from fastestimator.op.numpyop.multivariate import LongestMaxSize, PadIfNeeded
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop import Dice
from fastestimator.op.tensorop.loss import FocalLoss, MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Dice as DiceScore
from fastestimator.trace.trace import Trace


def collect_single_keypoint_ds(ds, cache_limit=None):
    images, keypoints, keypoint_bboxes = [], [], []
    for idx in range(len(ds)):
        item = ds[idx]
        for keypoint, keypoint_bbox in zip(item['keypoint'], item['keypoint_bbox']):
            images.append(item['image'])
            keypoints.append(keypoint)
            keypoint_bboxes.append(keypoint_bbox)
        if idx % 1000 == 0 or idx + 1 == len(ds):
            print("Loading data --- {} / {}".format(idx + 1, len(ds)))
        if cache_limit and idx + 1 == cache_limit:
            break
    return NumpyDataset(data={"image": images, "keypoint": keypoints, "keypoint_bbox": keypoint_bboxes})


class KeypointMask(NumpyOp):
    def forward(self, data, state):
        image, keypoint, kpscore = data
        height, width, _ = image.shape
        kp_mask = np.stack([self.gaussian(kp, ks, height, width) for kp, ks in zip(keypoint, kpscore)], axis=-1)
        return kp_mask

    def gaussian(self, kp, ks, height, width):
        x0, y0 = kp
        x0, y0 = int(x0), int(y0)
        mask = np.zeros((height, width), dtype="float32")
        if ks >= 1:
            y_min, y_max = max(y0 - 2, 0), min(y0 + 3, height)
            x_min, x_max = max(x0 - 2, 0), min(x0 + 3, width)
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    # only worry about the 5x5 around keypoint center
                    mask[yi, xi] = np.exp(-((xi - x0)**2 + (yi - y0)**2) / (2 * 1**2))
        return mask


class CropImageKeypoint(NumpyOp):
    def forward(self, data, state):
        image, keypoint_bbox, keypoint = data
        image = self._crop_image(image, keypoint_bbox)
        keypoints, kpscore = self._crop_keypoint(keypoint, keypoint_bbox)
        return image, keypoints, kpscore

    def _crop_image(self, image, bbox):
        x1, y1, box_w, box_h = bbox
        x2, y2 = x1 + box_w, y1 + box_h
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        image = image[y1:y2, x1:x2]
        return image

    def _crop_keypoint(self, keypoints, bbox):
        x1, y1, w, h = bbox
        kpscore = keypoints[:, -1]
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        kp_x = np.clip(keypoints[:, 0] - x1, a_min=0, a_max=w - 1)
        kp_y = np.clip(keypoints[:, 1] - y1, a_min=0, a_max=h - 1)
        keypoints = [(x, y) for x, y in zip(kp_x, kp_y)]
        return keypoints, kpscore


class KeypointAccuracy(Trace):
    def on_epoch_begin(self, data):
        self.TN, self.TP, self.FP, self.FN = 0, 0, 0, 0

    def on_batch_end(self, data):
        pred_mask = data["pred_mask"].numpy()
        gt_mask = data["kp_mask"].numpy()
        for pred, gt in zip(pred_mask, gt_mask):
            self.update_counts(pred, gt)

    def update_counts(self, pred, gt):
        num_channel = gt.shape[-1]
        for idx in range(num_channel):
            pred_s, gt_s = pred[..., idx], gt[..., idx]
            gt_center = np.array(center_of_mass(gt_s)) if gt_s.max() == 1 else None
            if gt_center is None:
                if pred_s.max() >= 0.5:
                    self.FP += 1
                else:
                    self.TN += 1
            elif pred_s.max() >= 0.5:
                # if no positive prediction and gt exists, then add 1 to false negative
                self.FN += 1
            else:
                pred_center = (np.median(np.where(pred_s == pred_s.max())[0]),
                               np.median(np.where(pred_s == pred_s.max())[1]))
                if np.linalg.norm(np.array(pred_center) - gt_center) > 7:
                    self.FP += 1
                else:
                    self.TP += 1

    def on_epoch_end(self, data):
        data.write_with_log("kp_acc", (self.TP + self.TN) / (self.TN + self.TP + self.FN + self.FP))
        data.write_with_log("FP", self.FP)
        data.write_with_log("TP", self.TP)
        data.write_with_log("FN", self.FN)
        data.write_with_log("TN", self.TN)


def get_estimator():
    train_ds, eval_ds = load_data(root_dir="/raid/shared_data", load_bboxes=False, load_keypoints=True, replacement=False)
    # train_ds, eval_ds = collect_single_keypoint_ds(train_ds), collect_single_keypoint_ds(eval_ds)
    train_ds = collect_single_keypoint_ds(train_ds)
    
    # pipeline = fe.Pipeline(
    #     train_data=train_ds,
    #     eval_data=eval_ds,
    #     batch_size=16,
    #     ops=[
    #         ReadImage(inputs="image", outputs="image"),
    #         CropImageKeypoint(inputs=("image", "keypoint_bbox", "keypoint"), outputs=("image", "keypoint", "kpscore")),
    #         LongestMaxSize(max_size=256, image_in="image", keypoints_in="keypoint", keypoint_params='xy'),
    #         PadIfNeeded(min_height=256,
    #                     min_width=256,
    #                     image_in="image",
    #                     keypoints_in="keypoint",
    #                     keypoint_params='xy',
    #                     border_mode=cv2.BORDER_CONSTANT,
    #                     value=0),
    #         KeypointMask(inputs=("image", "keypoint", "kpscore"), outputs="kp_mask"),
    #         Delete(keys=("keypoint", "keypoint_bbox", "kpscore"))
    #     ])
    # model = fe.build(model_fn=lambda: UNet(input_size=(256, 256, 3), output_channel=17),
    #                  optimizer_fn=lambda: tf.optimizers.Adam(1e-4))
    # network = fe.Network(ops=[
    #     ModelOp(inputs="image", model=model, outputs="pred_mask"),
    #     MeanSquaredError(inputs=("pred_mask", "kp_mask"), outputs="mse"),
    #     # Dice(inputs=("pred_mask", "kp_mask"), outputs="loss", negate=True, sample_average=True),
    #     # FocalLoss(inputs=("pred_mask", "kp_mask"), outputs="loss", normalize=False),
    #     UpdateOp(model=model, loss_name="mse")
    # ])
    # traces = [
    #     DiceScore(true_key="kp_mask", pred_key="pred_mask"),
    #     # KeypointAccuracy(inputs=("pred_mask", "kp_mask"), outputs=("kp_acc", "FP", "FN", "TP", "TN"), mode="eval")
    # ]
    # estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=5, traces=traces)
    # return estimator

if __name__ == "__main__":
    get_estimator()
