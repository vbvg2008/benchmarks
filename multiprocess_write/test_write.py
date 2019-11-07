import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor


def write_files(dir, n_files, start_idx, pid):
    for i in range(start_idx, start_idx + n_files):
        n_example = 0.0
        time_start = time.perf_counter()
        with open(os.path.join(dir, "file_{}.txt".format(i)), 'w') as writer:
            for j in range(100):
                writer.write(1024 * 1024 * "{}".format(j))
                n_example += 1
                z = 0
                while z < 100000:
                    z += 1
        z = 0
        while z < 1000000:
            z += 1
        print("Thread {}: {:.2f} records/sec".format(pid, n_example / (time.perf_counter() - time_start)))


def run():
    n_files_per_thread = 20
    data_dir = "/data/data/test"
    num_process = os.cpu_count() or 1
    os.makedirs(data_dir, exist_ok=True)
    futures = []
    with ProcessPoolExecutor(max_workers=num_process) as executor:
        file_idx_start = 0
        for i in range(num_process):
            futures.append(executor.submit(write_files, data_dir, n_files_per_thread, file_idx_start, i))
            file_idx_start += n_files_per_thread
        for future in futures:
            result = future.result()


if __name__ == '__main__':
    run()
