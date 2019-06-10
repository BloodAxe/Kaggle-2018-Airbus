import os

import cv2
import numpy as np
import pandas as pd

from skimage import measure
from skimage.morphology import dilation
from skimage import measure
from skimage.filters import median
from skimage.morphology import dilation, watershed, square, erosion, disk

from lib.dataset.common import rle2instancemask, get_rle_from_groundtruth, get_bboxes_from_groundtruth, read_train_image
from lib.dataset.rssd_dataset import instance_mask_to_rbboxes
from lib.dataset.segmentation import instance_mask_to_fg_and_edge_v2, instance_mask_to_fg_and_edge
from lib.visualizations import visualize_rbbox

data_dir = 'd:\\datasets\\airbus'


def test_mask():
    # id = 'aed55df18.jpg'
    id = 'dc6adaf6b.jpg'
    groundtruth = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))
    image = read_train_image(id, data_dir)
    rle = get_rle_from_groundtruth(groundtruth, id)
    label_image = rle2instancemask(rle)

    mask = instance_mask_to_fg_and_edge(label_image)

    dummy = np.zeros((label_image.shape[0], label_image.shape[1], 1), dtype=np.uint8)
    mask = np.concatenate([mask * 255, dummy], axis=2)
    cv2.imshow('mask (no edge)', (label_image > 0).astype(np.uint8) * 255)
    cv2.imshow('mask', mask)
    # cv2.waitKey(-1)

    rbboxes = instance_mask_to_rbboxes(label_image)
    for bbox in rbboxes:
        visualize_rbbox(image, bbox, (0, 255, 0))

    cv2.imshow('a', image)
    cv2.waitKey(-1)


def test_get_bboxes_from_groundtruth():
    groundtruth = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))
    bboxes = get_bboxes_from_groundtruth(groundtruth)
    ships = bboxes[bboxes[:, 2] > 0]

    print(len(groundtruth))
    print(len(bboxes))
    print(len(ships))
    print(ships[:, 2].mean())
    print(ships[:, 3].mean())
