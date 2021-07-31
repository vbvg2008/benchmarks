"""
git clone --recursive https://github.com/apache/tvm tvm
./build.sh conda_cuda100
create a container, go in container and navigate to the tvm folder
apt-get update
apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
modify config.cmake, set USE_CUDA, USE_LLVM, USE_CUDNN, USE_BLAS as openblas, MKL as ON
apt-get install clang-6.0 lldb-6.0 lld-6.0
install intel mkl https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-apt-repo.html
wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
apt-get update
apt-get install intel-mkl-64bit-2020.4-912
apt-get install libopenblas-dev
cd build
cmake ..
make -j4 (change 4 to other number if you have more cores)
cd python
pip install decorator attrs
python setup.py install
pip install pytest xgboost onnx
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
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner


def tune_kernels(tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


def tune_graph(graph, dshape, records, opt_sch_file, target, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {"input_1": dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


def evaluate_fn(module, data):
    module.set_input("input_1", data)
    module.run()
    output = module.get_output(0)
    return output


if __name__ == "__main__":
    onnx_model = onnx.load("/data/Xiaomeng/onnx/model.onnx")
    shape_dict = {"input_1": (1, 224, 224, 3)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = "llvm -mcpu=core-avx2 -libs=cblas"
    device = tvm.cpu()

    print("Tuning...")
    log_file = "tvm_cpu.log"
    graph_opt_sch_file = "resnet50_graph_opt.log"
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), ))
    tuning_option = {
        "log_filename":
        log_file,
        "tuner":
        "random",
        "early_stopping":
        None,
        "measure_option":
        autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True),
        ),
    }
    tune_kernels(tasks, **tuning_option)
    # tune_graph(mod["main"], (1, 224, 224, 3), log_file, graph_opt_sch_file, target=target)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(
                opt_level=2,
                required_pass=[
                    "FastMath", "FoldScaleAxis", "CanonicalizeOps", "CanonicalizeCast", "EliminateCommonSubexpr"
                ]):
            lib = relay.build_module.build(mod, target=target, params=params)
        # load params
        module = runtime.GraphModule(lib["default"](device))
        print("Evaluate...")
        sample_data = np.random.rand(1, 224, 224, 3).astype("float32")
        timeit(f=lambda: evaluate_fn(module, sample_data), num_runs=300)
