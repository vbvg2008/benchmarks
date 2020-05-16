import pdb

import cv2
import numpy as np
import tensorflow as tf


def load_and_preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (299, 299))
    img = img / 255
    img = img - 0.5
    img = img * 2
    return img


img = load_and_preprocess("image.png")
img = np.expand_dims(img, axis=0)

model = tf.keras.applications.InceptionV3(weights="imagenet")

results = model.predict(img)

pdb.set_trace()
