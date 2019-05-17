import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Input
import time

def my_model():
    inp = Input(shape=[28, 28])
    x = Flatten()(inp)
    x = Dense(128, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = my_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

idx = np.random.randint(0, 60000, 128)
imgs = train_images[idx]
label = train_labels[idx]

time_list = []
start_time = time.time()

for i in range(3281):
    if i % 100 == 0 and i >0:
        end_time = time.time()
        time_list.append(end_time - start_time)
        start_time = end_time
        print(i)
    # idx = np.random.randint(0, 60000, 128)
    # imgs = train_images[idx]
    # label = train_labels[idx]
    model.train_on_batch(imgs, label)

average_time = np.mean(time_list[4:])
example_per_sec = 100*128/average_time

print("training speed is %f examples/sec" % example_per_sec)

#example per second is 35494/sec