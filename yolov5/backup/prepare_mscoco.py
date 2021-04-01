import os
import pdb
import shutil

import cv2
from fastestimator.dataset.data import mscoco



def categoryid_2_classid(categoryid):
    categoryid_to_name = {
        "1": "person",
        "2": "bicycle",
        "3": "car",
        "4": "motorcycle",
        "5": "airplane",
        "6": "bus",
        "7": "train",
        "8": "truck",
        "9": "boat",
        "10": "traffic light",
        "11": "fire hydrant",
        "13": "stop sign",
        "14": "parking meter",
        "15": "bench",
        "16": "bird",
        "17": "cat",
        "18": "dog",
        "19": "horse",
        "20": "sheep",
        "21": "cow",
        "22": "elephant",
        "23": "bear",
        "24": "zebra",
        "25": "giraffe",
        "27": "backpack",
        "28": "umbrella",
        "31": "handbag",
        "32": "tie",
        "33": "suitcase",
        "34": "frisbee",
        "35": "skis",
        "36": "snowboard",
        "37": "sports ball",
        "38": "kite",
        "39": "baseball bat",
        "40": "baseball glove",
        "41": "skateboard",
        "42": "surfboard",
        "43": "tennis racket",
        "44": "bottle",
        "46": "wine glass",
        "47": "cup",
        "48": "fork",
        "49": "knife",
        "50": "spoon",
        "51": "bowl",
        "52": "banana",
        "53": "apple",
        "54": "sandwich",
        "55": "orange",
        "56": "broccoli",
        "57": "carrot",
        "58": "hot dog",
        "59": "pizza",
        "60": "donut",
        "61": "cake",
        "62": "chair",
        "63": "couch",
        "64": "potted plant",
        "65": "bed",
        "67": "dining table",
        "70": "toilet",
        "72": "tv",
        "73": "laptop",
        "74": "mouse",
        "75": "remote",
        "76": "keyboard",
        "77": "cell phone",
        "78": "microwave",
        "79": "oven",
        "80": "toaster",
        "81": "sink",
        "82": "refrigerator",
        "84": "book",
        "85": "clock",
        "86": "vase",
        "87": "scissors",
        "88": "teddy bear",
        "89": "hair drier",
        "90": "toothbrush"
    }
    class_names = [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush'
    ]
    return class_names.index(categoryid_to_name[str(categoryid)])


def generate_label_data(ds, output_path):
    label_paths = []
    bbox_count = []
    for idx in range(len(ds)):
        data = ds[idx]
        label_file = os.path.splitext(os.path.basename(data["image"]))[0] + ".txt"
        label_path = os.path.join(output_path, label_file)
        image = cv2.imread(data["image"])
        im_h, im_w = image.shape[0], image.shape[1]
        bbox_count.append(len(data["bbox"]))
        label_paths.append(label_path)
        with open(label_path, 'w') as f:
            for x1, y1, width, height, categoryid in data["bbox"]:
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                center_x = center_x / im_w
                center_y = center_y / im_h
                width = width / im_w
                height = height / im_h
                classid = categoryid_2_classid(categoryid)
                f.writelines("{} {} {} {} {}\n".format(classid, center_x, center_y, width, height))
        if idx % 100 == 0:
            print(idx)


def remove_extra_images(image_folder, label_folder, move_folder):
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")
        if not os.path.exists(label_path):
            shutil.move(os.path.join(image_folder, image_name), os.path.join(move_folder, image_name))
            print(os.path.join(move_folder, image_name))


if __name__ == "__main__":
    # train_ds, eval_ds = mscoco.load_data(root_dir="/data/data/public")
    # pdb.set_trace()

    # generate_label_data(train_ds, "/data/data/public/MSCOCO2017/labels/train2017")
    # generate_label_data(eval_ds, "/data/data/public/MSCOCO2017/labels/val2017")
    remove_extra_images("/data/data/public/COCO2017/MSCOCO2017/train2017",
                        "/data/data/public/MSCOCO2017/coco/labels/train2017",
                        "/data/data/public/COCO2017/extra_images/train2017")
    remove_extra_images("/data/data/public/COCO2017/MSCOCO2017/val2017",
                        "/data/data/public/MSCOCO2017/coco/labels/val2017",
                        "/data/data/public/COCO2017/extra_images/val2017")
