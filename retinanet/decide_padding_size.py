import time

import fastestimator as fe


def find_magic_numer(pipeline, mode, current_epoch=0, num_steps=1000, log_interval=100):
    pipeline.global_batch_multiplier = 1
    global_batch_size = pipeline.get_global_batch_size(current_epoch)
    pipeline.prepare()
    ds_iter = pipeline.dataset_schedule[mode].get_current_value(current_epoch)
    start = time.perf_counter()
    max_length = 0
    for idx in range(num_steps + 1):
        batch_data = next(ds_iter)
        length = batch_data["x1_gt"].shape[-1]
        if length > max_length:
            max_length = length
        if idx % log_interval == 0:
            if idx == 0:
                start = time.perf_counter()
            else:
                duration = time.perf_counter() - start
                example_per_sec = log_interval * global_batch_size / duration
                print("FastEstimator: Step: %d, Epoch: %d, Batch Size %d, Example/sec %.2f" %
                      (idx, current_epoch, global_batch_size, example_per_sec))
                print("magic number so far is {}".format(max_length))
                start = time.perf_counter()
    print("final magic number is {}".format(max_length))
    pipeline._reset()


if __name__ == "__main__":
    pipeline = fe.Pipeline(batch_size=16, data="/data/data/MSCOCO2017/retinanet_coco", padded_batch=True)
    find_magic_numer(pipeline, "train", num_steps=400)
