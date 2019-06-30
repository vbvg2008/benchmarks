import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

#read the batch prediction file
npz_file = np.load("epoch11.npz")
valid_outputs = npz_file["valid_outputs"]
cls_selected = npz_file["cls_selected"]
loc_selected = npz_file["loc_selected"]
image = npz_file["image"]
width = 128
height = 64

#class SaveBoundingImage(Trace):
#    def __init__(self, batch_idx=[0]):
#        self.batch_idx = batch_idx
#
#    def on_batch_end(self, mode, logs):
#        if mode == "eval" and logs["step"] in self.batch_idx:
#            cls_selected, loc_selected, valid_outputs = logs["prediction"] #cls_selected is [A], loc_selected is [A, 4], valida_outputs is [B]
#            image = np.array(logs["batch"]["image"])
#            cls_selected, loc_selected, valid_outputs = np.array(cls_selected), np.array(loc_selected), np.array(valid_outputs)
#            np.savez("epoch%d" % logs["epoch"], cls_selected=cls_selected, loc_selected=loc_selected, valid_outputs=valid_outputs, image=image)
#            print("saving predicted results to epoch%d" % logs["epoch"])

def display_img(idx):
    # Display the image
    plt.imshow(image[idx])
    num_bb = valid_outputs[idx]
    idx_start = sum(valid_outputs[:idx])
    for j in range(num_bb):
        x1 = int(np.round(loc_selected[idx_start+j, 0]*width))
        y1 = int(np.round(loc_selected[idx_start+j, 1]*height))
        x2 = int(np.round(loc_selected[idx_start+j, 2]*width))
        y2 = int(np.round(loc_selected[idx_start+j, 3]*height))
        plt.gca().add_patch(Rectangle((x1,y1),width=x2-x1, height=y2-y1, fill=False,color="r"))
        print("the label is %d" % cls_selected[idx_start+j])