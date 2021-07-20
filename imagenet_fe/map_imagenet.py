import json
import os
import pdb


def create_val_mapping_json(val_dir="/data/data/public/ImageNet/val"):
    dictionary = {}
    for folder in os.listdir(val_dir):
        for file in os.listdir(os.path.join(val_dir, folder)):
            dictionary[file] = folder

    with open("mapping.json", 'w') as f:
        json.dump(dictionary, f)
