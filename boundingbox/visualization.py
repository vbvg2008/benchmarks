import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

npz_file = np.load("epoch9.npz")
cls_selected = npz_file["cls_selected"]
loc_selected = npz_file["loc_selected"]
valid_outputs = npz_file["valid_outputs"]
image = npz_file["image"]

width = 128
height = 64

def display_img(idx):
    # Display the image
    plt.imshow(image[idx])
    num_bb = valid_outputs[idx]
    for j in range(num_bb):
        x1 = int(np.round(loc_selected[0, 0]*width))
        y1 = int(np.round(loc_selected[0, 1]*height))
        x2 = int(np.round(loc_selected[0, 2]*width))
        y2 = int(np.round(loc_selected[0, ]*height))
        plt.gca().add_patch(Rectangle((x1,y1),width=x2-x1, height=y2-y1, fill=False,color="r"))
        print("the label is %d" % cls_selected[0])

idx = 0
display_img(idx)