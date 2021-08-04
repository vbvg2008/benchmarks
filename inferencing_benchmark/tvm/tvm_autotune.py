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
pip install pytest xgboost tensorflow
"""
import argparse
import os
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from util import timeit

import tvm
import tvm.relay.testing
from tvm import auto_scheduler, autotvm, relay, te
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner
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


def tune_tasks(tasks,
               measure_option,
               tuner="xgb",
               n_trial=1000,
               early_stopping=None,
               log_filename="tuning.log",
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def run(d, hw):
    assert d in {"gpu", "cpu"}
    assert hw in {"p3", "p2", "pc"}
    shape_dict = {"input_1": (1, 3, 224, 224)}
    keras_resnet50 = my_resnet50()
    mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
    network = "resnet-50"
    batch_size = 1
    if d == "cpu":
        target = tvm.target.Target("llvm -mcpu=core-avx2")
        device = tvm.cpu()
    else:
        target = target = tvm.target.Target("cuda")
        device = tvm.cuda(0)
    log_file = "%s-%s-B%d-%s.log" % (network, hw, batch_size, target.kind.name)
    print("Tuning...")
    if not os.path.exists(log_file):
        tasks = autotvm.task.extract_from_program(mod["main"],
                                                  target=target,
                                                  params=params,
                                                  ops=(relay.op.get("nn.conv2d"), ))
        if d == "gpu":
            tuning_option = {
                "log_filename":
                log_file,
                "tuner":
                "xgb",
                "n_trial":
                2000,
                "early_stopping":
                600,
                "measure_option":
                autotvm.measure_option(builder=autotvm.LocalBuilder(timeout=10),
                                       runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150))
            }
        else:
            tuning_option = {
                "log_filename":
                log_file,
                "tuner":
                "random",
                "n_trial":
                2000,
                "early_stopping":
                600,
                "measure_option":
                autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True),
                ),
            }
        tune_tasks(tasks, **tuning_option)
    print("Compile...")
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    print("Evaluate...")
    module = graph_executor.GraphModule(lib["default"](device))
    sample_data = np.random.rand(1, 3, 224, 224).astype("float32")
    timeit(f=lambda: evaluate_fn(module, sample_data), num_runs=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True, help="cpu or gpu")
    parser.add_argument("--hw", type=str, required=True, help="p3, p2, or pc")
    args = parser.parse_args()
    run(args.d, args.hw)
