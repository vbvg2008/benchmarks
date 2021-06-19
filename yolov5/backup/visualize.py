import pdb
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_original_image(filename="datanew.npy", slice_idx=4):
    data = np.load("data.npy")
    image = data[slice_idx]
    image = np.transpose(image, axes=[1, 2, 0])
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.show()


def visualize_compare(filename1="data.npy", filename2="datanew.npy"):
    data1 = np.load(filename1)
    data1 = np.transpose(data1, axes=[0, 2, 3, 1])
    data2 = np.load(filename2)
    print(np.sum(data1 == 114))
    print(np.sum(data2 == 114))
    indexes1 = random.sample(list(range(64)), 4)
    indexes2 = random.sample(list(range(64)), 4)
    fig, axs = plt.subplots(2, 4)
    for i in range(4):
        axs[0, i].imshow(data1[indexes1[i]])
        axs[0, i].axis('off')
        axs[1, i].imshow(data2[indexes2[i]])
        axs[1, i].axis('off')
    plt.show()


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min +
                                                 w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color,
                  thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
        print("{}: {}".format(class_name, bbox))
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def visualize_image_with_box(images="image.npy", bboxes="bbox.npy"):
    category_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    category_id_to_name = {
        key: name
        for key, name in zip(range(80), category_names)
    }
    images = np.load(images)
    bboxes = np.load(bboxes)
    images = np.uint8(images * 255)
    num_images = images.shape[0]
    index = np.random.randint(0, num_images)
    image = images[index]
    bbox = bboxes[index]
    bbox = bbox[np.sum(bbox, axis=1) != 0]
    bbox_coco = [(x[0], x[1], x[2], x[3]) for x in bbox]
    category_ids = [int(x[4]) for x in bbox]
    visualize(image, bbox_coco, category_ids, category_id_to_name)


def visualize_image_with_box_official(images="image_official.npy",
                                      bboxes="bbox_official.npy"):
    category_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    category_id_to_name = {
        key: name
        for key, name in zip(range(80), category_names)
    }
    images = np.load(images)
    bboxes = np.load(bboxes)
    num_images = images.shape[0]
    index = np.random.randint(0, num_images)
    image = images[index]
    image = np.transpose(image, [1, 2, 0])
    bbox = bboxes[bboxes[:, 0] == index]
    height, width, _ = image.shape
    bbox_coco = [
        (x[2] * width - x[4] * width / 2, x[3] * height - x[5] * height / 2,
         x[4] * width, x[5] * height) for x in bbox
    ]
    category_ids = [int(x[1]) for x in bbox]
    visualize(image, bbox_coco, category_ids, category_id_to_name)


# visualize_image_with_box()
visualize_image_with_box_official()
