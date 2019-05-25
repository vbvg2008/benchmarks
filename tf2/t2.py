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

# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10, activation='softmax')

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)

model = MyModel()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

# model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

#metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  print(loss)
  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

#   for test_images, test_labels in test_ds:
#     test_step(test_images, test_labels)

#   template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#   print (template.format(epoch+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          test_loss.result(),
#                          test_accuracy.result()*100))
