"""
git clone --recursive https://github.com/apache/tvm tvm
./build.sh conda_cuda100
nvidia-docker run -it --name tvm_cblas -v /data:/data tvm.conda_cuda100:latest /bin/bash, go in container and navigate to the tvm folder
apt-get update
apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
modify config.cmake, set USE_CUDA, USE_LLVM, USE_CUDNN, CUBLAS as ON
apt-get install clang-6.0 lldb-6.0 lld-6.0
cd build
cmake ..
make -j4 (change 4 to other number if you have more cores)
cd python
pip install numpy decorator attrs
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
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor
from tvm.relay import data_dep_optimization as ddo


def evaluate_fn(module, data):
    module.set_input("data", data)
    module.run()
    output = module.get_output(0)
    return output


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size, ) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape


def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            ) for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)


if __name__ == "__main__":
    network = "resnet-50"
    use_sparse = False
    batch_size = 1
    layout = "NHWC"
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    dtype = "float32"
    log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
    print("Get model...")
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
    # print("Extract tasks...")
    # tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    # print("Tuning...")
    # run_tuning(tasks, task_weights, log_file=log_file)
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build_module.build(mod, target=target, params=params)
    device = tvm.cpu()
    # load params
    module = runtime.GraphModule(lib["default"](device))
    print("Evaluate...")
    sample_data = np.random.rand(1, 224, 224, 3).astype("float32")
    timeit(f=lambda: evaluate_fn(module, sample_data), num_runs=100)
