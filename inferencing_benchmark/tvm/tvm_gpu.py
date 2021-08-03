"""
git clone --recursive https://github.com/apache/tvm tvm
./build.sh conda_cuda100
create a container, go in container and navigate to the tvm folder
apt-get update
apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
modify config.cmake, set USE_CUDA, USE_LLVM, USE_CUDNN as ON
apt-get install clang-6.0 lldb-6.0 lld-6.0
cd build
cmake ..
make -j4 (change 4 to other number if you have more cores)
cd python
pip install numpy decorator attrs
python setup.py install
pip install pytest xgboost
"""
import os
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from util import timeit

import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor


def evaluate_fn(module, data):
    module.set_input("input_1", data)
    module.run()
    output = module.get_output(0)
    return output


def my_resnet50():
    inputs = layers.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
    x = backbone.outputs[0]
    outputs = layers.Dense(1000, activation='softmax')(x)
    # outputs = layers.Activation('softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
    if os.path.exists(log_file):
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


if __name__ == "__main__":
    shape_dict = {"input_1": (1, 3, 224, 224)}
    keras_resnet50 = my_resnet50()
    mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
    network = "resnet-50"
    batch_size = 1
    machine = "p3"
    target = tvm.target.Target("cuda")
    log_file = "%s-%s-B%d-%s.json" % (network, machine, batch_size, target.kind.name)

    print("Tuning...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    run_tuning(tasks, task_weights, log_file)
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    print("Evaluate...")
    device = tvm.cuda(0)
    module = graph_executor.GraphModule(lib["default"](device))
    sample_data = np.random.rand(1, 3, 224, 224).astype("float32")
    timeit(f=lambda: evaluate_fn(module, sample_data), num_runs=1000)
