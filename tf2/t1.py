from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, ELU, Conv2D, MaxPool2D
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def conv_block(self, x, filters):
        x = Conv2D(filters=filters, kernel_size=(5,5))(x)
        x = ELU()(x)
        x = MaxPool2D()(x)
        return x

    def dense_block(self, x, units):
        x = Dense(units)(x)
        x = ELU()(x)
        return x

    def call(self, x):
        x = self.conv_block(x, 32)
        x = self.conv_block(x, 64)
        x = Flatten()(x)
        x = self.dense_block(x, 256)
        x = self.dense_block(x, 128)
        x = Dense(10, activation='softmax')(x)
        return x

class MyModel2(Model):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.conv1 =  Conv2D(filters=32, kernel_size=(5,5))
        self.conv2 = Conv2D(filters=64, kernel_size=(5,5))
        self.d1 = Dense(256)
        self.d2 = Dense(128)
        self.d3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = ELU()(x)
        x = MaxPool2D()(x)
        x = self.conv2(x)
        x = ELU()(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = self.d1(x)
        x = ELU()(x)
        x = self.d2(x)
        x = ELU()(x)
        x = self.d3(x)
        return x

class MyModel3(Model):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def my_model():
    inp = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(5,5))(inp)
    x = ELU()(x)
    x = MaxPool2D()(x)
    x = Conv2D(filters=64, kernel_size=(5,5))(x)
    x = ELU()(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = ELU()(x)
    x = Dense(128)(x)
    x = ELU()(x)
    out = Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)

model = MyModel2()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)