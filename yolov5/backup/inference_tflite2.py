import pdb
import time

import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov5.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
total_time = []
num_trials = 100
for _ in range(num_trials):
    start = time.time()
    interpreter.invoke()
    total_time.append(time.time() - start)
print("Average Inferencing speed is {} ms with {} trials".format(np.mean(total_time[1:]) * 1000, num_trials))
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

# Yolov5 on CPU: 165ms/run
