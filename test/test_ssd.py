import cv2
import numpy as np
import torch
from skimage.color import label2rgb

from lib.dataset.ssd import instance_mask_to_bboxes
from lib.models.fpnssd512.box_coder import FPNSSDBoxCoder
from lib.visualizations import visualize_bbox


def test_ssd_synthetic():
    label_image = np.zeros((512, 512), dtype=np.uint8)

    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 100), (100, 20), 0)), 1).astype(int), (1, 1, 1))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100), (100, 20), 45)), 1).astype(int), (2, 2, 2))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 200), (100, 20), 90)), 1).astype(int), (3, 3, 3))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200), (100, 20), 135)), 1).astype(int), (4, 4, 4))

    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 100), (20, 100), 0)), 1).astype(int), (5, 5, 5))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 100), (20, 100), 45)), 1).astype(int), (6, 6, 6))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 200), (20, 100), 90)), 1).astype(int), (7, 7, 7))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 200), (20, 100), 135)), 1).astype(int), (8, 8, 8))

    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 100 + 200), (100, 20), 17)), 1).astype(int), (9, 9, 9))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100 + 200), (100, 20), 49)), 1).astype(int), (10, 10, 10))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 200 + 200), (100, 20), 99)), 1).astype(int), (11, 11, 11))
    # cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200 + 200), (100, 20), 165)), 1).astype(int), (12, 12, 12))

    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[25, 90], [125, 90], [125, 110], [25, 110]]), 1), (1, 1, 1))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[10, 400], [70, 400], [70, 420], [10, 420]]), 1), (3, 3, 3))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[100, 100], [200, 110], [200, 140], [110, 130]]), 1), (4, 4, 4))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[300, 200], [400, 210], [410, 330], [310, 330]]), 1), (5, 5, 5))

    image = (label2rgb(label_image, bg_label=0) * 255).astype(np.uint8)

    # Test what happens if we rotate
    # image = np.rot90(image).copy()
    # label_image = np.rot90(label_image).copy()

    bboxes = instance_mask_to_bboxes(label_image)
    print(bboxes)

    labels = np.zeros(len(bboxes), dtype=np.intp)

    box_coder = FPNSSDBoxCoder()

    loc_targets, cls_targets, anchors = box_coder.encode(torch.from_numpy(bboxes).float(),
                                                         torch.from_numpy(labels), return_anchors=True)
    print(loc_targets.shape, cls_targets.shape)

    cls_targets_one_hot = np.eye(2)[cls_targets]
    print(cls_targets_one_hot.shape)

    dec_boxes, dec_labels, dec_scores = box_coder.decode(loc_targets, torch.from_numpy(cls_targets_one_hot))
    print(dec_boxes)

    for bbox in dec_boxes.numpy():
        visualize_bbox(image, bbox, (255, 0, 255), thickness=3)

    for bbox in bboxes:
        visualize_bbox(image, bbox, (0, 255, 0), thickness=1)

    for bbox in anchors.numpy():
        visualize_bbox(image, bbox, (255, 255, 255), thickness=1)

    cv2.imshow('overlays', image)
    # cv2.imshow('anchors', anchors)
    cv2.waitKey(-1)
