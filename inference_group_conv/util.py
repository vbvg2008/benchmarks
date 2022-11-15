import time

import numpy as np


def timeit(f, num_runs=100, warmup=True):
    times = []
    if warmup:
        f()  # first call the function for a good luck
    for _ in range(num_runs):
        start = time.time()
        f()
        times.append(time.time() - start)
    print("After {} runs, average running time is {} ms".format(num_runs, 1000 * np.mean(times)))
