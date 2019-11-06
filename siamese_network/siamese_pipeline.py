import os

import fastestimator as fe
from fastestimator.dataset.mnist import load_data
from fastestimator.op import NumpyOp, TensorOp
from fastestimator.op.numpyop import ImageReader
from fastestimator.op.tensorop import Minmax, ScalarFilter


class GetSimilarity(TensorOp):
    def forward(self, data, state):
        y_a, y_b = data
        if y_a == y_b:
            similarity = 1.0
        else:
            similarity = 0.0
        return similarity


def get_pipeline():
    train_csv, _, path = load_data()
    # for writer, we can create two unpaired dataset (see tutorial 10)
    writer = fe.RecordWriter(
        save_dir=os.path.join(path, "siamese_tfrecord"),
        train_data=(train_csv, train_csv),
        ops=([ImageReader(inputs="x", outputs="x_a", parent_path=path), NumpyOp(inputs="y", outputs="y_a")],
             [ImageReader(inputs="x", outputs="x_b", parent_path=path), NumpyOp(inputs="y", outputs="y_b")]),
        write_feature=(["x_a", "y_a"], ["x_b", "y_b"]))

    # in pipeline, we will randomly pair them, then scale the chances of different category with filterop (see tutorial 6)
    pipeline = fe.Pipeline(
        data=writer,
        batch_size=32,
        ops=[
            Minmax(inputs="x_a", outputs="x_a"),
            Minmax(inputs="x_b", outputs="x_b"),
            GetSimilarity(inputs=("y_a", "y_b"), outputs="similarity")
            # ScalarFilter(inputs="similarity", filter_value=0.0,
            #              keep_prob=1 / 9)  # 9/10 chance to pick differnet category, rescaling the ratio to be 1:1
        ])
    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.benchmark()
    # result = pipeline.show_results()
    # print("the label from A is {}".format(result[0]["y_a"]))
    # print("the label from B is {}".format(result[0]["y_b"]))
    # print("the similarity between A and B is {}".format(result[0]["similarity"]))
