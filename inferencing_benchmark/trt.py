"""
docker pull nvcr.io/nvidia/tensorrt:21.07-py3

first follow the onnx tutorial and create onnx model

next, generate the trt engine
trtexec --onnx=/data/Xiaomeng/onnx/model.onnx --saveEngine=resnet_engine.trt  --explicitBatch

next, run the python script
"""
import pdb
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from util import timeit

f = open("/data/Xiaomeng/trt_v100/resnet_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
output = np.empty([1, 1000], dtype="float32")  # Need to set output dtype to FP16 to enable FP16

input_batch = np.random.rand(1, 224, 224, 3).astype("float32")

# Allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()


def predict(batch):  # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()

    return output


timeit(lambda: predict(input_batch))
