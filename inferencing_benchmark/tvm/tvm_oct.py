"""
given the wheel file,

1. use fe container
2. apt-get install libssl-dev zlib1g-dev libopenblas-dev build-essential
3. apt-get install libffi-dev
4. python3.7 -m pip install
"""
import pdb

import numpy as np
import Res50NX_2_broadwell
from util import timeit


def evaluate_fn(model, data):
    outputs = model.run(data)
    return outputs[0]


model = Res50NX_2_broadwell.OctomizedModel()

# Create random inputs to the model. This example code assumes a single-input
# model -- please adjust for your own purposes.
idict = model.get_input_dict()
iname = list(idict.keys())[0]
ishape = idict[iname]["shape"]
idtype = idict[iname]["dtype"]
sample_data = np.random.random(ishape).astype(idtype)
timeit(f=lambda: evaluate_fn(model, sample_data), num_runs=500)
