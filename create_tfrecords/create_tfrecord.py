from fastestimator.util.tfrecord import TFRecorder
from fastestimator.pipeline.dynamic.preprocess import ImageReader, Resize

tfrecorder = TFRecorder(train_data="train.csv", 
                        validation_data="val.csv",
                        feature_name=["image", "label"], 
                        transform_dataset=[[ImageReader(),Resize(size=[256,256]) ],[]])

tfrecorder.create_tfrecord("/data/data/Other/ImageNet_tfrecord")