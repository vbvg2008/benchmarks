import pdb
import pickle

import numpy as np

import fastestimator as fe
from fastestimator.architecture.tensorflow import LeNet

with open("batch32_1.pkl", 'rb') as f:
    gradients_all_gpus = pickle.load(f)

model_begin = LeNet()
model_begin.load_weights("begin.h5")
begin_varables = model_begin.trainable_variables

model_after_update = LeNet()
model_after_update.load_weights("after_update.h5")
after_varables = model_after_update.trainable_variables

for idx, _ in enumerate(gradients_all_gpus):
    var_length = begin_varables[idx].shape[0]
    combined_gradient = (gradients_all_gpus[idx][:var_length] + gradients_all_gpus[idx][var_length:2 * var_length] +
                         gradients_all_gpus[idx][2 * var_length:3 * var_length] +
                         gradients_all_gpus[idx][3 * var_length:]) / 4
    actual_gradient = (begin_varables[idx] - after_varables[idx]) / 0.01
    actual_gradient = actual_gradient.numpy()
    difference = actual_gradient - combined_gradient
    mean1 = np.mean(np.abs(combined_gradient))
    mean2 = np.mean(np.abs(actual_gradient))
    difference = np.max(np.abs(difference))
    print(difference / mean1)
    print("===========================")
