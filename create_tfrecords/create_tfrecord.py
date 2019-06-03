from fastestimator.util.tfrecord import TFRecorder
from fastestimator.pipeline.dynamic.preprocess import ImageReader, Resize
import time


tfrecorder = TFRecorder(train_data="train.csv", 
                        validation_data="val.csv",
                        feature_name=["image", "label"], 
                        transform_dataset=[[ImageReader(),Resize(size=[256,256]) ],[]],
                        compression="GZIP")
start = time.time()
tfrecorder.create_tfrecord("/data/data/ImageNet_tfrecord")
print("creating ImageNet tfrecord spent %f seconds" % (time.time() - start))