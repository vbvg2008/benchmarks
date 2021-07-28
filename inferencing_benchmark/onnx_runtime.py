"""
pip install -U tf2onnx
pip install onnxruntime
python -m tf2onnx.convert --saved-model /data/saved_model/1/ --output model.onnx
INFO - Model inputs: ['input_1']
INFO - Model outputs: ['dense']
"""
import pdb

import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("/data/Xiaomeng/onnx/model.onnx")
got = session.run(None, {'input_1': np.random.rand(1, 224, 224, 3).astype("float32")})
pdb.set_trace()
print(got[0])
