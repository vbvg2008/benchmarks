"""
docker pull openvino/ubuntu18_dev

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
  --data_type=FP32 \
  --saved_model_dir /data/Xiaomeng/saved_model/1 \
  --input_shape [1,224,224,3] \
  --output_dir /data/Xiaomeng/openvino

"""

import numpy as np
from openvino.inference_engine import IECore

model_xml = "/data/Xiaomeng/openvino/saved_model.xml"
model_bin = "/data/Xiaomeng/openvino/saved_model.bin"
ie_core = IECore()
net = ie_core.read_network(model=model_xml, weights=model_bin)
exec_net = ie_core.load_network(network=net, device_name="CPU")
res = exec_net.infer({'input_1': np.random.rand(1, 3, 224, 224).astype("float32")})
final = res["StatefulPartitionedCall/model/dense/Softmax"]
