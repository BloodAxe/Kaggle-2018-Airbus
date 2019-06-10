import numpy as np
import torch

from lib.common.box_util import box_nms_fast, box_nms


def test_nms():
    bboxes = [[10, 10, 20, 30],
              [9, 11, 21, 29],
              [11, 7, 19, 31]]

    scores = torch.from_numpy(np.array([0.7, 1, 0.6]))
    bboxes = torch.from_numpy(np.array(bboxes))

    nms1 = box_nms(bboxes, scores)
    nms2 = box_nms_fast(bboxes, scores)
    print(nms1)
    print(nms2)
