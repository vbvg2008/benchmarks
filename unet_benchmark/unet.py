import os
import pdb
import subprocess
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def timeit(f, num_runs=1000, warmup=True):
    times = []
    if warmup:
        f()  # first call the function for a good luck
    for _ in range(num_runs):
        start = time.time()
        f()
        times.append(time.time() - start)
    print("After {} runs, average running time is {} ms".format(num_runs, 1000 * np.mean(times)))


def apply_conv(encoder_in, c_encoder, c_tasks, tasks_in=None):
    conv_layer = Conv2D(c_encoder, 3, activation='relu', padding='same')
    conv_layer.trainable = False
    encoder_out = conv_layer(encoder_in)
    if tasks_in:
        tasks_out = [concatenate([encoder_in, task_in], axis=-1) for task_in in tasks_in]
    else:
        tasks_out = [encoder_in for _ in range(len(c_tasks))]
    tasks_out = [Conv2D(c, 3, activation='relu', padding='same')(task_out) for c, task_out in zip(c_tasks, tasks_out)]
    return encoder_out, tasks_out


def apply_pool(encoder_in, tasks_in):
    encoder_out = MaxPooling2D()(encoder_in)
    tasks_out = [MaxPooling2D()(x) for x in tasks_in]
    return encoder_out, tasks_out


def apply_upsample(encoder_in, tasks_in):
    encoder_out = UpSampling2D()(encoder_in)
    tasks_out = [UpSampling2D()(x) for x in tasks_in]
    return encoder_out, tasks_out


def apply_concat(encoder_in, tasks_in):
    tasks_out = [concatenate([encoder_in, task_in]) for task_in in tasks_in]
    return tasks_out


def new_channel(base_c, c_news):
    return tuple(base_c // 64 * c for c in c_news)


def encoders(inputs, c_news):
    conv1, conv1_tasks = apply_conv(inputs, 64, new_channel(64, c_news))
    conv1, conv1_tasks = apply_conv(conv1, 64, new_channel(64, c_news), tasks_in=conv1_tasks)
    pool1, pool1_tasks = apply_pool(conv1, conv1_tasks)

    conv2, conv2_tasks = apply_conv(pool1, 128, new_channel(128, c_news), tasks_in=pool1_tasks)
    conv2, conv2_tasks = apply_conv(conv2, 128, new_channel(128, c_news), tasks_in=conv2_tasks)
    pool2, pool2_tasks = apply_pool(conv2, conv2_tasks)

    conv3, conv3_tasks = apply_conv(pool2, 256, new_channel(256, c_news), tasks_in=pool2_tasks)
    conv3, conv3_tasks = apply_conv(conv3, 256, new_channel(256, c_news), tasks_in=conv3_tasks)
    pool3, pool3_tasks = apply_pool(conv3, conv3_tasks)

    conv4, conv4_tasks = apply_conv(pool3, 512, new_channel(512, c_news), tasks_in=pool3_tasks)
    conv4, conv4_tasks = apply_conv(conv4, 512, new_channel(512, c_news), tasks_in=conv4_tasks)
    pool4, pool4_tasks = apply_pool(conv4, conv4_tasks)

    conv5, conv5_tasks = apply_conv(pool4, 1024, new_channel(1024, c_news), tasks_in=pool4_tasks)
    conv5, conv5_tasks = apply_conv(conv5, 1024, new_channel(1024, c_news), tasks_in=conv5_tasks)

    conv1_tasks = apply_concat(conv1, conv1_tasks)
    conv2_tasks = apply_concat(conv2, conv2_tasks)
    conv3_tasks = apply_concat(conv3, conv3_tasks)
    conv4_tasks = apply_concat(conv4, conv4_tasks)
    conv5_tasks = apply_concat(conv5, conv5_tasks)
    return conv1_tasks, conv2_tasks, conv3_tasks, conv4_tasks, conv5_tasks


def decoder(conv1, conv2, conv3, conv4, conv5, output_channel=1, dec=32):
    up6 = Conv2D(max(16, dec), 3, activation='relu', padding='same')(UpSampling2D()(conv5))
    merge6 = concatenate([conv4, up6], axis=-1)
    conv6 = Conv2D(max(16, dec), 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(max(16, dec), 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(max(16, dec // 2), 3, activation='relu', padding='same')(UpSampling2D()(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(max(16, dec // 2), 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(max(16, dec // 2), 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(max(16, dec // 4), 3, activation='relu', padding='same')(UpSampling2D()(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(max(16, dec // 4), 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(max(16, dec // 4), 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(max(16, dec // 8), 3, activation='relu', padding='same')(UpSampling2D()(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(max(16, dec // 8), 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(max(16, dec // 8), 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(output_channel, 1, activation='sigmoid')(conv9)
    return conv10


def UNet(input_shape=(128, 128, 1),
         output_channel=(1, 1, 1, 1, 1, 1, 1),
         c_news=(4, 4, 4, 4, 4, 4, 4),
         decs=(32, 32, 32, 32, 32, 32, 32)):
    inputs = Input(shape=input_shape)
    conv1_tasks, conv2_tasks, conv3_tasks, conv4_tasks, conv5_tasks = encoders(inputs, c_news)
    output_tasks = [
        decoder(conv1, conv2, conv3, conv4, conv5, c, dec) for conv1,
        conv2,
        conv3,
        conv4,
        conv5,
        c,
        dec in zip(conv1_tasks, conv2_tasks, conv3_tasks, conv4_tasks, conv5_tasks, output_channel, decs)
    ]
    # task_models = [tf.keras.Model(inputs=inputs, outputs=output_task) for output_task in output_tasks]

    output_overall = concatenate(output_tasks, axis=-1)
    overall_model = tf.keras.Model(inputs=inputs, outputs=output_overall)
    return overall_model


class GPUMemory(object):
    def __enter__(self) -> None:
        self.process = subprocess.Popen(
            "while true; do nvidia-smi -i 0 --query-gpu=memory.used  --format=csv,noheader; sleep .1; done",
            shell=True,
            stdout=subprocess.PIPE)
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.process.kill()
        outputs = self.process.stdout.read().decode("utf-8").strip().split("\n")
        self.max_usage = max([int(x.split(" ")[0]) for x in outputs])
        self.unit = outputs[0].split(" ")[1]


@tf.function
def single_inference(model, data):
    output = model(data, training=False)
    return output


if __name__ == "__main__":
    output_channel = (1, 1, 1, 1, 1, 1, 1)
    c_news = (32, 32, 32, 32, 32, 32, 32)
    decs = (64, 64, 64, 64, 64, 64, 64)
    model = UNet(input_shape=(512, 512, 1), output_channel=output_channel, c_news=c_news, decs=decs)

    data = np.random.rand(1, 512, 512, 1).astype("float32")
    with GPUMemory() as gpu_m:
        timeit(f=lambda: single_inference(model=model, data=data))
    print("max memory usage: {} MiB".format(gpu_m.max_usage))
