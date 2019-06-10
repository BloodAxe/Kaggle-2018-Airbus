import math
import os

import cv2
import torch
import numpy as np
from skimage.color import label2rgb
from lib.dataset import rssd_dataset as D
from lib.common.box_util import meshgrid, meshgrid_numpy
from lib.common.rbox_util import angle_diff, compute_overlap
from lib.dataset.common import read_train_image, get_rle_from_groundtruth, rle2instancemask, get_train_test_split_for_fold, all_test_ids
from lib.dataset.rssd_dataset import instance_mask_to_rbboxes
from lib.models.fpnssd512.box_coder import FPNSSDBoxCoder
from lib.models.rssd.rsdd import RSSD, RSSDBoxCoder
from lib.models.retinanet.rretina_net import RRNBoxCoder
from lib.visualizations import visualize_rbbox
from train_rssd import get_transform


def test_sdd_box_coder():
    box_coder = FPNSSDBoxCoder()
    #
    boxes = [
        [20, 40, 80, 100],
        [200, 4, 300, 200],
        [100, 100, 160, 200],
        [50, 90, 175, 300],
    ]

    labels = [
        0,
        0,
        0,
        0
    ]

    loc_targets, cls_targets = box_coder.encode(torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    dec_boxes, dec_labels, dec_scores = box_coder.decode(loc_targets, cls_targets)
    print(dec_boxes, dec_labels, dec_scores)


def test_overlap():
    # Same boxes
    np.testing.assert_almost_equal(compute_overlap((0, 0, 10, 5, 20), (0, 0, 10, 5, 20)), 1., decimal=4)

    # Same same boxes, second one is flipped by 180 CW
    np.testing.assert_almost_equal(compute_overlap(((0, 0, 10, 5, 20)), (0, 0, 10, 5, 20 + 180)), 1., decimal=4)

    # Same same boxes, second one is flipped by 180 CCW
    np.testing.assert_almost_equal(compute_overlap(((0, 0, 10, 5, 20)), (0, 0, 10, 5, 20 - 180)), 1., decimal=4)

    # Same same boxes, second one is flipped by 180 CCW
    np.testing.assert_almost_equal(compute_overlap(((0, 0, 10, 5, 20)), (0, 0, 5, 10, 20 - 90)), 1., decimal=4)

    # Same same boxes, second one is flipped by 180 CCW
    np.testing.assert_almost_equal(compute_overlap(((0, 0, 10, 5, 20)), (0, 0, 5, 10, 20 + 90)), 1., decimal=4)

    # 0.5 IOU
    np.testing.assert_almost_equal(compute_overlap(((0, 0, 10, 5, 20)), (0, 0, 5, 5, 20)), 0.5, decimal=4)


def test_anchors_count():
    n = len(FPNSSDBoxCoder().anchor_boxes)
    print('Total number of anchor boxes SSD512', n)

    box_coder = RSSDBoxCoder(512, 512)
    n = len(box_coder.anchor_boxes)
    print('Total number of anchor boxes in RSSD512', n)

    box_coder = RSSDBoxCoder(768, 768)
    n = len(box_coder.anchor_boxes)
    print('Total number of anchor boxes in RSSD768', n)


def test_draw_rsdd_bboxes():
    box_coder = RSSDBoxCoder(768, 768)
    anchors = box_coder._get_anchor_wht()

    n = len(FPNSSDBoxCoder().anchor_boxes)
    print('Total number of anchor boxes SSD', len(box_coder.anchor_boxes))

    n = len(box_coder.anchor_boxes)
    print('Total number of anchor boxes in RSSD', len(box_coder.anchor_boxes))

    for i, wht_fm in enumerate(anchors):

        image = np.zeros((box_coder.image_height, box_coder.image_width, 3), dtype=np.uint8)

        for wht in wht_fm:
            rbox = [box_coder.image_height // 2, box_coder.image_width // 2, wht[0], wht[1], wht[2]]
            visualize_rbbox(image, rbox, (i * 28, 255, 0), thickness=1)

        cv2.imshow("Image" + str(i), image)
    cv2.waitKey(-1)


def test_draw_sersdd_bboxes():
    box_coder = RRNBoxCoder(768, 768)
    anchors = box_coder._get_anchor_wht()

    n = len(box_coder.anchor_boxes)
    print('Total number of anchor boxes in SERSSDBoxCoder', len(box_coder.anchor_boxes))

    for i, (wht_fm, fm_size) in enumerate(zip(anchors, box_coder.feature_map_sizes)):
        image = np.zeros((box_coder.image_height, box_coder.image_width, 3), dtype=np.uint8)

        grid_size = box_coder.input_size / fm_size
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])

        xy = meshgrid_numpy(fm_w, fm_h).astype(np.float32) + 0.5
        xy = (xy * grid_size).reshape(-1, 2)

        for p in xy:
            p = p.astype(int)
            cv2.circle(image, tuple(p), 2, (200, 200, 200), cv2.FILLED)

            for wht in wht_fm:
                a = math.sqrt(wht[0] * wht[1])
                w = int(a)
                h = int(a)

                cv2.rectangle(image, (p[0] - w // 2, p[1] - h // 2), (p[0] + w // 2, p[1] + h // 2), (100, 100, 100))

                # rbox = [p[0], p[1], wht[0], wht[1], wht[2]]
                # visualize_rbbox(image, rbox, (i * 28, 255, 0), thickness=1)

        for wht in wht_fm:
            rbox = [box_coder.image_height // 2, box_coder.image_width // 2, wht[0], wht[1], wht[2]]
            visualize_rbbox(image, rbox, (i * 28, 255, 0), thickness=1)

        cv2.imshow("Image" + str(i), image)
    cv2.waitKey(-1)


def test_rsdd_box_coder():
    box_coder = RSSDBoxCoder(512, 512)
    # print('RSSDBoxCoder')
    # print(box_coder.anchor_boxes)

    #
    boxes = np.array([
        [20, 40, 80, 10, 15],
        [200, 4, 100, 20, 150],
        [100, 100, 60, 20, 95],
        [50, 90, 75, 30, 5],
    ], dtype=np.float32)

    labels = np.array([
        0,
        0,
        0,
        0
    ])

    loc_targets, cls_targets = box_coder.encode(boxes, labels)
    print(loc_targets.shape, cls_targets.shape)

    cls_targets_one_hot = np.eye(2)[cls_targets]
    print(cls_targets_one_hot.shape)

    dec_boxes, dec_labels, dec_scores = box_coder.decode(loc_targets, cls_targets_one_hot)
    print(dec_boxes, dec_labels, dec_scores)


import pandas as pd

data_dir = 'd:\\datasets\\airbus'


def test_rssd_encoding():
    id = 'dc6adaf6b.jpg'
    groundtruth = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))
    image = read_train_image(id, data_dir)
    rle = get_rle_from_groundtruth(groundtruth, id)
    label_image = rle2instancemask(rle)

    # Test what happens if we rotate
    # image = np.rot90(image).copy()
    # label_image = np.rot90(label_image).copy()

    rbboxes = instance_mask_to_rbboxes(label_image)
    print('Instances', rbboxes)

    labels = np.zeros(len(rbboxes), dtype=np.intp)

    box_coder = RSSDBoxCoder(768, 778)

    loc_targets, cls_targets = box_coder.encode(rbboxes, labels)
    print(loc_targets.shape, cls_targets.shape)

    cls_targets_one_hot = np.eye(2)[cls_targets]
    print(cls_targets_one_hot.shape)

    dec_boxes, dec_labels, dec_scores = box_coder.decode(loc_targets, cls_targets_one_hot)
    print(dec_boxes)

    for bbox in dec_boxes:
        visualize_rbbox(image, bbox, (255, 0, 0), thickness=3)

    for bbox in rbboxes:
        visualize_rbbox(image, bbox, (0, 255, 0), thickness=1)

    cv2.imshow('a', image)
    cv2.waitKey(-1)


def test_angle_diff():
    np.testing.assert_almost_equal(angle_diff(0., 179.), 1.)
    np.testing.assert_almost_equal(angle_diff(1., -1.), 2.)
    np.testing.assert_almost_equal(angle_diff(0., 90.), 90.)
    np.testing.assert_almost_equal(angle_diff(-90., 90.), 0.)


def test_meshgrid():
    fm_h = 8
    fm_w = 8
    xy = meshgrid_numpy(fm_w, fm_h) + 0.5
    print(xy)

    xy2 = meshgrid(fm_w, fm_h).float() + 0.5
    print(xy2)

    np.testing.assert_equal(xy, xy2.numpy())


def test_rssd_synthetic():
    label_image = np.zeros((768, 768), dtype=np.uint8)

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 100), (100, 20), 0)), 1).astype(int), (1, 1, 1))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100), (100, 20), 45)), 1).astype(int), (2, 2, 2))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 200), (100, 20), 90)), 1).astype(int), (3, 3, 3))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200), (100, 20), 135)), 1).astype(int), (4, 4, 4))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 100), (20, 100), 0)), 1).astype(int), (5, 5, 5))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 100), (20, 100), 45)), 1).astype(int), (6, 6, 6))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 200), (20, 100), 90)), 1).astype(int), (7, 7, 7))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 200), (20, 100), 135)), 1).astype(int), (8, 8, 8))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 105 + 200), (100, 20), 45. / 2)), 1).astype(int), (9, 9, 9))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100 + 200), (16, 4), 49)), 1).astype(int), (10, 10, 10))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100 + 210), (100, 20), 49)), 1).astype(int), (11, 11, 11))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200 + 200), (100, 20), 165)), 1).astype(int), (12, 12, 12))

    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[25, 90], [125, 90], [125, 110], [25, 110]]), 1), (1, 1, 1))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[10, 400], [70, 400], [70, 420], [10, 420]]), 1), (3, 3, 3))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[100, 100], [200, 110], [200, 140], [110, 130]]), 1), (4, 4, 4))
    # cv2.fillConvexPoly(label_image, np.expand_dims(np.array([[300, 200], [400, 210], [410, 330], [310, 330]]), 1), (5, 5, 5))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300, 300), (10, 6), 49)), 1).astype(int), (13, 13, 13))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300 + 50, 300), (200, 40), 90)), 1).astype(int), (14, 14, 14))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300 + 70, 300), (100, 20), 90)), 1).astype(int), (15, 15, 15))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((500, 500), (2, 3), 9)), 1).astype(int), (16, 16, 16))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((510, 500), (2, 3), 19)), 1).astype(int), (17, 17, 17))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((540, 500), (2, 3), 29)), 1).astype(int), (18, 18, 18))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((560, 500), (2, 3), 39)), 1).astype(int), (19, 19, 19))

    image = (label2rgb(label_image, bg_label=0) * 255).astype(np.uint8)

    # Test what happens if we rotate
    # image = np.rot90(image).copy()
    # label_image = np.rot90(label_image).copy()

    rbboxes = instance_mask_to_rbboxes(label_image)
    print(rbboxes)

    labels = np.zeros(len(rbboxes), dtype=np.intp)

    box_coder = RSSDBoxCoder(768, 768)

    loc_targets, cls_targets, anchors = box_coder.encode(rbboxes,
                                                         labels,
                                                         return_anchors=True)
    print(loc_targets.shape, cls_targets.shape)

    cls_targets_one_hot = np.eye(2)[cls_targets]
    print(cls_targets_one_hot.shape)

    dec_boxes, dec_labels, dec_scores = box_coder.decode(loc_targets,
                                                         torch.from_numpy(cls_targets_one_hot))
    print(dec_boxes)
    print('Total anchors', len(anchors))

    for bbox in anchors:
        visualize_rbbox(image, bbox, (255, 255, 255), thickness=1)

    for bbox in dec_boxes:
        visualize_rbbox(image, bbox, (255, 0, 255), thickness=3)

    for bbox in rbboxes:
        visualize_rbbox(image, bbox, (0, 255, 0), thickness=1)

    cv2.imshow('overlays', image)
    cv2.waitKey(-1)


def test_rrssd_box_coder_synthetic():
    label_image = np.zeros((768, 768), dtype=np.uint8)

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 100), (100, 20), 0)), 1).astype(int), (1, 1, 1))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100), (100, 20), 45)), 1).astype(int), (2, 2, 2))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 200), (100, 20), 90)), 1).astype(int), (3, 3, 3))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200), (100, 20), 135)), 1).astype(int), (4, 4, 4))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 100), (20, 100), 0)), 1).astype(int), (5, 5, 5))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 100), (20, 100), 45)), 1).astype(int), (6, 6, 6))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100 + 200, 200), (20, 100), 90)), 1).astype(int), (7, 7, 7))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200 + 200, 200), (20, 100), 135)), 1).astype(int), (8, 8, 8))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((100, 105 + 200), (100, 20), 45. / 2)), 1).astype(int), (9, 9, 9))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100 + 200), (16, 4), 49)), 1).astype(int), (10, 10, 10))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 100 + 210), (100, 20), 49)), 1).astype(int), (11, 11, 11))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((200, 200 + 200), (100, 20), 165)), 1).astype(int), (12, 12, 12))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300, 300), (10, 6), 49)), 1).astype(int), (13, 13, 13))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300 + 50, 300), (200, 40), 90)), 1).astype(int), (14, 14, 14))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((300 + 70, 300), (100, 20), 90)), 1).astype(int), (15, 15, 15))

    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((500, 500), (2, 3), 9)), 1).astype(int), (16, 16, 16))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((510, 500), (2, 3), 19)), 1).astype(int), (17, 17, 17))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((540, 500), (4, 6), 29)), 1).astype(int), (18, 18, 18))
    cv2.fillConvexPoly(label_image, np.expand_dims(cv2.boxPoints(((560, 500), (2, 3), 39)), 1).astype(int), (19, 19, 19))

    image = (label2rgb(label_image, bg_label=0) * 255).astype(np.uint8)

    # Test what happens if we rotate
    # image = np.rot90(image).copy()
    # label_image = np.rot90(label_image).copy()

    rbboxes = instance_mask_to_rbboxes(label_image)
    print(rbboxes)

    labels = np.zeros(len(rbboxes), dtype=np.intp)

    box_coder = RRNBoxCoder(768, 768)

    loc_targets, cls_targets, anchors = box_coder.encode([],
                                                         [],
                                                         return_anchors=True)


    loc_targets, cls_targets, anchors = box_coder.encode(rbboxes,
                                                         labels,
                                                         return_anchors=True)
    print(loc_targets.shape, cls_targets.shape, anchors.shape)
    print('Object anchors', (cls_targets>0).sum())
    print('Background anchors', (cls_targets==0).sum())
    print('Ignore anchors', (cls_targets==-1).sum())

    # cls_targets = np.expand_dims(cls_targets)
    # print(cls_targets_one_hot.shape)

    dec_boxes, dec_scores = box_coder.decode(loc_targets, cls_targets)
    print(dec_boxes)
    print('Total anchors', len(anchors))
    for bbox in dec_boxes:
        visualize_rbbox(image, bbox, (255, 0, 255), thickness=3)

    for bbox in rbboxes:
        visualize_rbbox(image, bbox, (0, 255, 0), thickness=1)

    for bbox in anchors:
        visualize_rbbox(image, bbox, (255, 255, 255), thickness=1)

    cv2.imshow('overlays', image)
    cv2.waitKey(-1)


def test_can_make_off_predictions():
    box_coder = RSSDBoxCoder(768, 768)
    _, valid_ids = get_train_test_split_for_fold(fold=0)
    validset_full = D.RSSDDataset(sample_ids=valid_ids,
                                  data_dir=data_dir,
                                  transform=get_transform(training=False, width=768, height=768),
                                  box_coder=box_coder)

    model = RSSD(validset_full.get_num_classes()).cuda()

    oof_predictions = model.predict_as_csv(validset_full)
    oof_predictions.to_csv('unittest_oof_predictions.csv')


def test_can_make_test_predictions():
    box_coder = RSSDBoxCoder(768, 768)
    testset_full = D.RSSDDataset(sample_ids=all_test_ids(data_dir),
                                 test=True,
                                 data_dir=data_dir,
                                 transform=get_transform(training=False, width=768, height=768),
                                 box_coder=box_coder)
    model = RSSD(testset_full.get_num_classes()).cuda()
    test_predictions = model.predict_as_csv(testset_full)
    test_predictions.to_csv('unittest_test_predictions.csv')
