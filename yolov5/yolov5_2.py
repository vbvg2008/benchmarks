import pdb

import numpy as np
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)

# Images
# img1 = Image.open('zidane.jpg')

# inputs = np.array(img1)
# inputs = np.transpose(inputs, [2, 0, 1])
# pdb.set_trace()
# # Inference
# result = model([inputs])
inputs = torch.rand(3, 720, 1280)
pred = model([inputs])
pdb.set_trace()
