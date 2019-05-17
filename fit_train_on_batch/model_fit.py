import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input


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
model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels), batch_size=128)


#43478 examples/sec