import pdb

import numpy as np

import fastestimator as fe

x_pos = list(range(1000))
y_pos = [1] * 1000
ds_pos = fe.dataset.NumpyDataset(data={"x": x_pos, "y": y_pos})

x_neg = list(range(1000, 1300))
y_neg = [0] * 300
ds_neg = fe.dataset.NumpyDataset(data={"x": x_neg, "y": y_neg})

x_unid = list(range(1300, 1600))
y_unid = [2] * 300
ds_unid = fe.dataset.NumpyDataset(data={"x": x_unid, "y": y_unid})

train_ds = fe.dataset.BatchDataset(datasets=[ds_pos, ds_neg, ds_unid], num_samples=[10, 4, 4])
pipeline = fe.Pipeline(train_data=train_ds)

for epoch in range(1, 10):
    example = []
    loader = pipeline.get_loader(mode="train", epoch=epoch)
    for data in loader:
        labels = list(data["y"].numpy())
        assert labels.count(1) == 10
        assert labels.count(0) == 4
        assert labels.count(2) == 4
        example.extend(data["x"].numpy())
    print("========epoch {}=========".format(epoch))
    print("histogram:")
    print(np.histogram(example)[0])

    print("number of unique examples in the epoch:")
    print(np.unique(example).shape[0])
