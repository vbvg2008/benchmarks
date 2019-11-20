import os
import random
import tempfile

import pandas as pd
import tensorflow as tf

import fastestimator as fe
from fastestimator.op.numpyop import ImageReader, Resize
from fastestimator.op.tensorop import ModelOp, Scale, SparseCategoricalCrossentropy
from fastestimator.trace import Accuracy, ModelSaver


def prepare_imagenet_csv(data_folder):
    modes = ["train", "val"]
    label_map = {}
    csv_paths = []
    for mode in modes:
        assert os.path.exists(os.path.join(data_folder, mode)), "cannot find {} folder in {}".format(mode, data_folder)
        csv_path = os.path.join(data_folder, mode + ".csv")
        if not os.path.exists(csv_path):
            folder_mode = os.path.join(data_folder, mode)
            class_ids = os.listdir(folder_mode)
            images = []
            labels = []
            for idx, class_id in enumerate(class_ids):
                if idx % 100 == 0:
                    print("preparing csv data for {}, progress: {:.1%}".format(mode, idx / len(class_ids)))
                folder_single_class = os.path.join(folder_mode, class_id)
                if class_id not in label_map:
                    label_map[class_id] = idx
                image_names = os.listdir(folder_single_class)
                for image_name in image_names:
                    images.append(os.path.join(folder_single_class, image_name))
                    labels.append(label_map[class_id])
            zipped_list = list(zip(images, labels))
            random.shuffle(zipped_list)
            df = pd.DataFrame(zipped_list, columns=["image", "label"])
            df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    return csv_paths[0], csv_paths[1]


def get_estimator(data_folder="/data/data/ImageNet2012", model_dir=tempfile.mkdtemp()):
    #step 0: make sure the data is downloaded to data_folder
    train_csv, val_csv = prepare_imagenet_csv(data_folder)
    #step 1: pipeline
    writer = fe.RecordWriter(save_dir=os.path.join(data_folder, "tfrecords"),
                             train_data=train_csv,
                             validation_data=val_csv,
                             ops=[ImageReader(inputs="image"), Resize(target_size=(299, 299), outputs="image")],
                             compression="GZIP")

    pipeline = fe.Pipeline(data=writer, batch_size=64, ops=Scale(inputs="image", scalar=1.0 / 255, outputs="image"))
    # step 2: Network
    model = fe.build(model_def=lambda: tf.keras.applications.InceptionV3(weights=None),
                     model_name="inceptionv3",
                     optimizer="adam",
                     loss_name="final_loss")
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="image", outputs="y_pred"),
        SparseCategoricalCrossentropy(inputs=("label", "y_pred"), outputs="final_loss")
    ])
    # step 3: Estimator
    traces = [
        Accuracy(true_key="label", pred_key="y_pred", output_name='acc'),
        ModelSaver(model_name="inceptionv3", save_dir=model_dir, save_best=True)
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=20, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
