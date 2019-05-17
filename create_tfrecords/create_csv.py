import os
import pandas as pd
import random

data_dir = "/data/data/Other/ImageNet/train"
file_name = "val.csv"


folder_names = os.listdir(data_dir)
images = []
labels = []

for i in range(len(folder_names)):
    if i % 100 == 0:
        print(i)
    folder_name = folder_names[i]
    folder_path = os.path.join(data_dir, folder_name)
    image_names = os.listdir(folder_path)
    for image_name in image_names:
        images.append(os.path.join(folder_path, image_name))
        labels.append(i)
    
print(len(images))
print(len(labels))

# zipped_list = list(zip(images,labels))
# random.shuffle(zipped_list)

# df = pd.DataFrame(zipped_list, columns = ["image", "label"])
# df.to_csv(file_name, index=False)

