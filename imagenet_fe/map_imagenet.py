import json
import os
import pdb
import tarfile
from shutil import copyfile


def create_val_mapping_json(val_dir="/data/data/public/ImageNet/val"):
    dictionary = {}
    for folder in os.listdir(val_dir):
        for file in os.listdir(os.path.join(val_dir, folder)):
            dictionary[file] = folder

    with open("mapping.json", 'w') as f:
        json.dump(dictionary, f)


def create_val_dirs(mapping_file, val_image_dir, val_dir):
    with open(mapping_file, 'r') as f:
        file_mapping = json.load(f)
    files = os.listdir(val_image_dir)

    for idx, file in enumerate(files):
        source_path = os.path.join(val_image_dir, file)
        target_folder = file_mapping[file]
        target_folder_path = os.path.join(val_dir, target_folder)
        if not os.path.exists(target_folder_path):
            os.mkdir(target_folder_path)
        target_path = os.path.join(target_folder_path, file)
        copyfile(source_path, target_path)
        if idx % 100 ==0:
            print(idx)


# create_val_dirs(mapping_file="mapping.json", val_image_dir="/raid/shared_data/ImageNet2012/val_images/", val_dir="/raid/shared_data/ImageNet2012/val/")

def extract_train_files(train_tar_dir, train_dir):
    tar_files = os.listdir(train_tar_dir)
    for idx, tar_file in enumerate(tar_files):
        folder_name = os.path.splitext(tar_file)[0]
        tar_path = os.path.join(train_tar_dir, tar_file)
        extract_path = os.path.join(train_dir, folder_name)
        if not os.path.exists(extract_path):
            os.mkdir(extract_path)
            # extract
            with tarfile.open(tar_path) as tar:
                tar.extractall(extract_path)
            if idx % 10 == 0:
                print(idx)

extract_train_files(train_tar_dir="/data/shared_data/ImageNet2012/train_tars/", train_dir="/data/shared_data/ImageNet2012/train/")
