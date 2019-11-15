import time

import tensorflow as tf
from torch.utils.data import DataLoader, Dataset

from skimage.feature import local_binary_pattern


class MnistDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = {"x": self.x[idx], "y": self.y[idx]}
        return self.transform_data(data)

    def transform_data(self, data):
        data["x"] = local_binary_pattern(data["x"], 24, 3)
        return data


if __name__ == "__main__":

    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    ds = MnistDataset(x=x_train, y=y_train)
    dataloader = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=4)
    ds_iter = iter(dataloader)
    tic = time.perf_counter()
    for idx in range(1850):
        batch_data = next(ds_iter)
        if idx > 0 and idx % 100 == 0:
            elapse = time.perf_counter() - tic
            example_sec = 100 * 32 / elapse
            print("step: {}, image/sec: {}".format(idx, example_sec))
            tic = time.perf_counter()
