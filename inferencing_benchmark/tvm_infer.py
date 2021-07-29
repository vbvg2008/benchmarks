"""
git clone --recursive https://github.com/apache/tvm tvm
./build.sh conda_cuda100
create a container, go in container and navigate to the tvm folder
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
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
pip install pytest
pip install xgboost
"""
import os
import pdb

import numpy as np
from util import timeit

import onnx
import tvm
import tvm.contrib.graph_executor as runtime
import tvm.relay as relay
import tvm.relay.testing
from tvm import autotvm, relay, te
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
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


if __name__ == "__main__":
    onnx_model = onnx.load("/data/Xiaomeng/onnx/model.onnx")
    shape_dict = {"input_1": (1, 224, 224, 3)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = "cuda"
    device = tvm.cuda(0)
    # target = "llvm"
    # device = tvm.cpu()

    print("Tuning...")
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), ))

    tuning_option = {
        "log_filename":
        "resnet50.log",
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

    tune_tasks(tasks, **tuning_option)
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        intrp = relay.build_module.create_executor("graph", mod, device, target)

    print("Evaluate...")
    sample_data = np.random.rand(1, 224, 224, 3).astype("float32")
    eval_fun = intrp.evaluate()
    timeit(f=lambda: eval_fun(tvm.nd.array(sample_data), **params).numpy(), num_runs=100)
