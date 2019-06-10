import cv2
import numpy as np
import torch

from lib.common.torch_utils import to_numpy


def from_opencv_rbox(rbox):
    (cx, cy), (w, h), t = rbox
    if w < h:
        w, h = h, w
        t = t - 90
    t = angle_to_180(t)
    assert w >= h
    assert 0 <= t < 180
    return [cx, cy, w, h, t]

    # corners = cv2.boxPoints(rbox)
    # v1 = corners[0] - corners[3]
    # v2 = corners[0] - corners[1]
    # d1 = np.linalg.norm(v1)
    # d2 = np.linalg.norm(v2)
    #
    # if d1 > d2:
    #     w = d1
    #     h = d2
    #     t = np.arctan2(v1[1], v1[0])
    # else:
    #     w = d2
    #     h = d1
    #     t = np.arctan2(v2[1], v2[0])
    #
    # (cx, cy), _, _ = rbox
    # t = 360 + t

    # assert 0 <= t < 180


def to_opencv_rbox(rbox):
    cx, cy, w, h, t = rbox
    # t = t - 360

    # if t > 90:
    # w, h = h, w
    # t -= 90

    return (cx, cy), (w, h), t


def rbbox2corners(rbox):
    """
    Computes 4 corners of the oriented bounding rectangle
    :param rbox: (xywht)
    :return: np.ndarray [4,2]
    """
    rrect = to_opencv_rbox(rbox)
    points = cv2.boxPoints(rrect)
    return points


def compute_overlap(box1, box2, distance_range=None, angle_range=None):
    # if angle_range is not None and angle_diff(box2[4], box1[4]) > angle_range:
    #     return 0

    # if distance_range is not None:
    #     sq_distance_range = distance_range * distance_range
    #     dv = box1[:2] - box2[:2]
    #     squared_distance = np.inner(dv, dv)  # Euclidean distance
    #     if squared_distance > sq_distance_range:
    #         return 0

    retval, intersectingRegion = cv2.rotatedRectangleIntersection(to_opencv_rbox(box1), to_opencv_rbox(box2))
    if retval == cv2.INTERSECT_NONE:
        return float(0)

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    intersection_area = cv2.contourArea(intersectingRegion, True)
    iou = float(intersection_area) / (area1 + area2 - intersection_area)
    return min(1., max(0., iou))


def angle_diff(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    d1 = angle_to_180(x - y)
    d2 = angle_to_180(y - x)
    return np.minimum(d1, d2)

    # x = np.deg2rad(x)
    # y = np.deg2rad(y)
    #
    # d1 = np.arccos(np.clip(np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), a_min=-1, a_max=1))
    # d2 = np.arccos(np.clip(np.cos(x + np.pi) * np.cos(y) + np.sin(x + np.pi) * np.sin(y), a_min=-1, a_max=1))
    # d = np.minimum(d1, d2)
    # return np.rad2deg(d)


def rbbox_iou(prior_boxes: np.ndarray, gt_boxes: np.ndarray):
    '''Compute the intersection over union of two set of oriented boxes.

    The box order must be (x, y, w, h, t).

    Args:
      prior_boxes: (np.ndarray) Prior (anchor) bounding boxes, sized [N,5].
      gt_boxes: (np.ndarray) Ground-truth bounding boxes, sized [M,5].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    return_torch = False
    if isinstance(prior_boxes, torch.Tensor):
        return_torch = True

    prior_boxes = to_numpy(prior_boxes)
    gt_boxes = to_numpy(gt_boxes)

    N = len(prior_boxes)
    M = len(gt_boxes)

    iou = np.zeros((N, M), dtype=np.float32)
    for j, box2 in enumerate(gt_boxes):
        iou[:, j] = intersection_area_one2many(box2, prior_boxes)

    if return_torch:
        iou = torch.from_numpy(iou)
    return iou


def intersection_area_one2many(box1, bboxes2, max_area_ratio=None):
    # If a single element we should expand first dim to pretend 2D array
    if len(bboxes2.shape) == 1:
        bboxes2 = np.expand_dims(bboxes2, 0)

    cx, cy, w, h, tt = box1
    spatial_dist = (bboxes2[:, 0] - cx) ** 2 + (bboxes2[:, 1] - cy) ** 2
    angle_dist = angle_diff(bboxes2[:, 4], tt)

    box1_area = box1[2] * box1[3]
    area = (bboxes2[:, 2] * bboxes2[:, 3])

    mask = (spatial_dist < w * w) & (angle_dist < 23)

    if max_area_ratio is not None:
        min_area = box1_area / max_area_ratio
        max_area = box1_area * max_area_ratio
        mask = mask & (area >= min_area) & (area <= max_area)

    iou = np.zeros(len(bboxes2), dtype=np.float32)
    iou[mask] = np.array([compute_overlap(box1, box2) for box2 in bboxes2[mask]], dtype=np.float32)

    return iou


def rbbox_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (np.ndarray) bounding boxes, sized [N,4].
      scores: (np.ndarray) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    # if there are no boxes, return an empty list
    if len(bboxes) == 0:
        return []

    if len(bboxes.shape) == 1:
        return [0]

    return_torch = False
    if isinstance(bboxes, torch.Tensor):
        return_torch = True

    # initialize the list of picked indexes
    pick = []

    # sort the bounding boxes by the confidence score
    idxs = np.argsort(scores)
    # print('NMS', len(scores))

    # corners = np.array([rbbox2corners(box) for box in bboxes])
    # area = bboxes[:, 2] * bboxes[:, 3]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compute the ratio of overlap
        overlap = intersection_area_one2many(bboxes[i], bboxes[idxs[:last]], max_area_ratio=2.)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
        # print('Size of idxs after nms iteration', len(idxs))

    pick = np.array(pick)
    if return_torch:
        pick = torch.from_numpy(pick)
    return pick


def angle_to_180(angle_in_degrees):
    if isinstance(angle_in_degrees, np.ndarray):
        angle_in_degrees[angle_in_degrees >= 180.] -= 180.
        angle_in_degrees[angle_in_degrees < 0.] += 180.

    if np.isscalar(angle_in_degrees):
        if angle_in_degrees >= 180.:
            angle_in_degrees -= 180.
        if angle_in_degrees < 0.:
            angle_in_degrees += 180.
        assert 0 <= angle_in_degrees < 180

    return angle_in_degrees


def clip_to_tan_range(angle_in_degrees):
    while (angle_in_degrees >= 90).any():
        angle_in_degrees[angle_in_degrees >= 90] -= 180

    while (angle_in_degrees <= -90).any():
        angle_in_degrees[angle_in_degrees <= -90] += 180

    # print(angle_in_degrees.min(), angle_in_degrees.max())
    return angle_in_degrees
