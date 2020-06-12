import time

import numpy as np


def minmax(data):
    data_min, data_max = np.min(data), np.max(data)
    data = (data - data_min) / (data_max - data_min)
    return data


if __name__ == "__main__":
    all_times = []
    for _ in range(10):
        data = np.random.randint(100, size=(1000, 1000, 10))
        begin_time = time.time()
        data = minmax(data)
        end_time = time.time()
        all_times.append(end_time - begin_time)
    print("after {} trials, the minmax took {} seconds in average".format(len(all_times), np.mean(all_times)))
