import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

npz_file = np.load("epoch0.npz")
cls_selected = npz_file["cls_selected"]
loc_selected = npz_file["loc_selected"]
valid_outputs = npz_file["valid_outputs"]
image = npz_file["image"]

image_size = 64
idx = 6
# Display the image
plt.imshow(image[idx])
num_bb = valid_outputs[idx]
for j in range(num_bb):
    x1,y1,x2,y2 = tuple(np.int32(np.round(loc_selected[10*idx+j]*image_size)))
    plt.gca().add_patch(Rectangle((x1,y1),width=x2-x1, height=y2-y1, fill=False))