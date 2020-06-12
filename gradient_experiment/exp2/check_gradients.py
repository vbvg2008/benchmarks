import pdb
import pickle

import numpy as np

with open("batch16_1.pkl", 'rb') as f:
    batch16_1 = pickle.load(f)

with open("batch16_2.pkl", 'rb') as f:
    batch16_2 = pickle.load(f)

with open("batch32_1.pkl", 'rb') as f:
    batch32_1 = pickle.load(f)

for idx, actual_gradient in enumerate(batch32_1):
    commutative_gradient = (batch16_1[idx] + batch16_2[idx]) / 2
    difference = actual_gradient - commutative_gradient
    mean1 = np.mean(np.abs(commutative_gradient))
    mean2 = np.mean(np.abs(actual_gradient))
    difference = np.max(np.abs(difference))
    print(difference / mean1)
    print("===========================")
