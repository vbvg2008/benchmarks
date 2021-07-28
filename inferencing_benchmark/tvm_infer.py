"""
git clone --recursive https://github.com/apache/tvm tvm
./build.sh conda_cuda100
"""
import pdb

import numpy as np

import onnx
import tvm
import tvm.relay as relay
from tvm import te

if __name__ == "__main__":
    onnx_model = onnx.load("/data/Xiaomeng/onnx/model.onnx")
    shape_dict = {"input_1": (1, 224, 224, 3)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    pdb.set_trace()
