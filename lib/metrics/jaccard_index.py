from tensorboardX import SummaryWriter
from torch import Tensor
import numpy as np

from lib.common.torch_utils import to_numpy


class JaccardIndex:
    def __init__(self, channel, threshold=None):
        self.channel = channel
        self.threshold = threshold
        self.scores_per_image = []

    def reset(self):
        self.scores_per_image = []

    def update(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_true.size(0)

        y_pred = y_pred[:, self.channel, ...].detach().view(batch_size, -1)
        y_true = y_true[:, self.channel, ...].detach().view(batch_size, -1)

        if self.threshold is not None:
            y_pred = y_pred > float(self.threshold)

        y_true = y_true.float()
        y_pred = y_pred.float()

        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        iou = intersection / (union - intersection + 1e-7)

        iou = iou[y_true.sum(dim=1) > 0]  # IoU defined only for non-empty masks
        self.scores_per_image.extend(to_numpy(iou))

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        if len(self.scores_per_image) == 0:
            return 0
        return np.mean(self.scores_per_image)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix + '/value', self.value(), step)
        saver.add_histogram(prefix + '/histogram', np.array(self.scores_per_image), step)
