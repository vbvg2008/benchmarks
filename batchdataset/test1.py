import pdb

import fastestimator as fe
from fastestimator.dataset import BatchDataset, LabeledDirDataset
from fastestimator.dataset.op_dataset import OpDataset
from fastestimator.op import NumpyOp
from fastestimator.op.numpyop.univariate import Minmax, ReadImage

dataset1 = LabeledDirDataset(root_dir="/data/data/test_mnist1", data_key="x", label_key="y")
dataset2 = LabeledDirDataset(root_dir="/data/data/test_mnist2", data_key="x", label_key="y")

batch_ds = BatchDataset(datasets=[dataset1, dataset2], num_samples=[4, 4])
# batch_ds2 = batch_ds.split(0.5)

pipeline = fe.Pipeline(train_data=batch_ds, ops=[ReadImage(inputs="x", outputs="x"), Minmax(inputs="x", outputs="x")])

data = pipeline.get_results()
pdb.set_trace()
