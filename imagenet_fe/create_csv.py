import pdb
import shutil

import pandas as pd

import fastestimator as fe
from fastestimator.dataset import CSVDataset


def create_new_files(csv_ds, mode):
    pos_image, pos_mask, pos_label = [], [], []
    neg_image, neg_mask, neg_label = [], [], []
    for i in range(len(csv_ds)):
        data = csv_ds[i]
        if data["label"] == 0:
            neg_image.append(data["image"])
            neg_mask.append(data["mask"])
            neg_label.append(data["label"])
        elif data["label"] == 1:
            pos_image.append(data["image"])
            pos_mask.append(data["mask"])
            pos_label.append(data["label"])
        else:
            raise ValueError("found unknown datasource {}".format(i))
    pos_df = pd.DataFrame(data={"image": pos_image, "mask": pos_mask, "label": pos_label})
    neg_df = pd.DataFrame(data={"image": neg_image, "mask": neg_mask, "label": neg_label})
    pos_df.to_csv(mode + "_positive.csv", index=False)
    neg_df.to_csv(mode + "_negative.csv", index=False)


def get_estimator():
    train_data = CSVDataset(file_path="/data/data/gehc/XRay/PTX/train.csv")
    eval_data = CSVDataset(file_path="/data/data/gehc/XRay/PTX/eval.csv")
    create_new_files(train_data, mode="train")
    create_new_files(eval_data, mode="eval")
    return None


if __name__ == "__main__":
    est = get_estimator()
    # est.fit()
