from tensorboardX import SummaryWriter
from torch import Tensor
import numpy as np


# helper function to calculate IoU
from lib.common.box_util import change_box_order
from lib.dataset.ssd import SSD_BBOXES_KEY, SSD_LABELS_KEY
from lib.models.fpnssd512.box_coder import FPNSSDBoxCoder
from lib.common.torch_utils import one_hot, to_numpy


def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    # assert w1 * h1 > 0
    # assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    assert np.isfinite(area1)
    assert np.isfinite(area2)
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


class BBoxMeanAP:
    def __init__(self, threshold=0.5):
        self.scores_per_image = []
        self.threshold = threshold
        self.box_coder = FPNSSDBoxCoder()

    def reset(self):
        self.scores_per_image = []

    def update(self, y_pred: Tensor, y_true: Tensor):
        true_ssd_bboxes = y_true[SSD_BBOXES_KEY].detach().cpu()
        pred_ssd_bboxes = y_pred[SSD_BBOXES_KEY].detach().cpu()
        pred_classes = y_pred[SSD_LABELS_KEY].detach().cpu()
        true_classes = y_true[SSD_LABELS_KEY].detach().cpu()

        pred_classes = pred_classes.softmax(dim=2)
        true_classes = one_hot(true_classes,
                               num_classes=pred_classes.size(2))

        for pred_loc, pred_cls, true_loc, true_cls in zip(pred_ssd_bboxes, pred_classes, true_ssd_bboxes, true_classes):
            pred_bboxes, _, pred_conf = self.box_coder.decode(pred_loc, pred_cls)
            true_bboxes, _, _ = self.box_coder.decode(true_loc, true_cls)

            true_bboxes = change_box_order(true_bboxes, 'xyxy2xywh')
            pred_bboxes = change_box_order(pred_bboxes, 'xyxy2xywh')

            true_bboxes = to_numpy(true_bboxes)
            pred_bboxes = to_numpy(pred_bboxes)
            pred_conf = to_numpy(pred_conf)

            if len(true_bboxes) == 0:
                continue

            if len(pred_bboxes) == 0:
                score = 0
            else:
                score = map_iou(true_bboxes, pred_bboxes, pred_conf)

            self.scores_per_image.append(score)

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        if len(self.scores_per_image) == 0:
            return 0
        return np.mean(self.scores_per_image)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        if len(self.scores_per_image) > 0:
            saver.add_scalar(prefix + '/value', self.value(), step)
            saver.add_histogram(prefix + '/histogram', np.array(self.scores_per_image), step)
