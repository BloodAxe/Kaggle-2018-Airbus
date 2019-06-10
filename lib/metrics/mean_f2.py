import json

import cv2
import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX.x2num import make_np
from torch import Tensor

from lib.common.torch_utils import one_hot, to_numpy
from lib.dataset.common import SSD_LABELS_KEY, SSD_BBOXES_KEY
from lib.visualizations import draw_rbboxes, visualize_rbbox

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def iou(y_true_img, y_pred_img):
    intersection = (y_pred_img * y_true_img).sum()
    union = y_pred_img.sum() + y_true_img.sum()
    jac = float(intersection / (union - intersection + 1e-7))
    return jac


def compute_f2(masks_true, masks_pred):
    if np.sum(masks_true) == 0:
        return float(np.sum(masks_pred) == 0)

    ious = []
    mp_idx_found = []
    for mt in masks_true:
        for mp_idx, mp in enumerate(masks_pred):
            if mp_idx not in mp_idx_found:
                cur_iou = iou(mt, mp)
                if cur_iou > 0.5:
                    ious.append(cur_iou)
                    mp_idx_found.append(mp_idx)
                    break
    f2_total = 0
    for th in thresholds:
        tp = sum([iou > th for iou in ious])
        fn = len(masks_true) - tp
        fp = len(masks_pred) - tp
        f2_total += (5 * tp) / (5 * tp + 4 * fn + fp)

    return f2_total / len(thresholds)


class F2ScoreFromLabelImage:
    """
        Computes F2 Score at different intersection over union (IoU) thresholds.
        This metric assumes that both y_pred, y_true are label images
    """

    def __init__(self, prob_to_pred_threshold=0.5):
        super().__init__()
        self.thresholds = np.array(thresholds)
        self.f2_scores = None
        self.prob_to_pred_threshold = prob_to_pred_threshold

        self.reset()

    def reset(self):
        self.f2_scores = []

    def update(self, y_pred, y_true):
        y_pred = self.expand_label_image(to_numpy(y_pred))
        y_true = self.expand_label_image(to_numpy(y_true))

        f2 = compute_f2(y_pred, y_true)
        self.f2_scores.append(f2)

    def expand_label_image(self, image):
        layers = []
        for id in range(1, image.max() + 1):
            layers.append(image == id)
        return np.array(layers, dtype=np.uint8)

    def value(self):
        return np.mean(self.f2_scores)

    def __str__(self):
        return '%.4f' % self.value()

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        if len(self.f2_scores) > 0:
            saver.add_scalar(prefix + '/value', self.value(), step)
            saver.add_histogram(prefix + '/histogram', np.array(self.f2_scores), step)


class F2ScoreFromRSSD:
    """
        Computes F2 Score from RSSD predictions
    """

    def __init__(self, box_coder):
        super().__init__()
        self.base_metric = F2ScoreFromLabelImage()
        self.box_coder = box_coder
        self.image_size = box_coder.image_height, box_coder.image_width

    def reset(self):
        self.base_metric.reset()

    def update(self, y_pred, y_true):

        num_classes = y_pred[SSD_BBOXES_KEY].size(2)

        true_bboxes = to_numpy(y_true[SSD_BBOXES_KEY])
        pred_bboxes = to_numpy(y_pred[SSD_BBOXES_KEY])
        pred_classes = to_numpy(y_pred[SSD_LABELS_KEY].detach().squeeze())
        true_classes = to_numpy(y_true[SSD_LABELS_KEY].detach().cpu())

        for pred_loc, pred_cls, true_loc, true_cls in zip(pred_bboxes, pred_classes, true_bboxes, true_classes):
            y_pred = self.rssd_predictions_to_ship_mask(pred_loc, pred_cls)
            y_true = self.rssd_predictions_to_ship_mask(true_loc, true_cls)
            self.base_metric.update(y_pred, y_true)

    def rssd_predictions_to_ship_mask(self, boxes, labels):
        boxes, scores = self.box_coder.decode(boxes, labels)
        mask = np.zeros(self.image_size, dtype=np.uint16)  # Just in case we have more than 255 ships in one image
        for i, bbox in enumerate(boxes):
            visualize_rbbox(mask, bbox, (i + 1, i + 1, i + 1), thickness=cv2.FILLED)
        return mask

    def value(self):
        return self.base_metric.value()

    def __str__(self):
        return '%.4f' % self.value()

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        self.base_metric.log_to_tensorboard(saver, prefix, step)
