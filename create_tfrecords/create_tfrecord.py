from fastestimator.util.tfrecord import TFRecorder
from fastestimator.pipeline.dynamic.preprocess import ImageReader, Resize
import time


tfrecorder = TFRecorder(train_data="train.csv",
                        validation_data="val.csv",
                        feature_name=["image", "label"],
                        transform_dataset=[[ImageReader(),Resize(size=[299,299]) ],[]])
start = time.time()
tfrecorder.create_tfrecord("/data/data/public/ImageNet2012/Tfrecord")
print("creating ImageNet tfrecord spent %f seconds" % (time.time() - start))

