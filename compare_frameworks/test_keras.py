import datetime

import tensorflow as tf
from test_base import batch_size, epochs, my_model_tf, x_eval, x_train, y_eval, y_train

startTime = datetime.datetime.now()

model = my_model_tf()

model.compile(optimizer=tf.optimizers.Adam(1e-4), metrics=["accuracy"], loss="categorical_crossentropy")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_eval, y_eval))

print("Elapsed seconds {}".format(datetime.datetime.now() - startTime))
