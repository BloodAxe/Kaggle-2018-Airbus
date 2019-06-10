from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter

from torch import Tensor


class ClassificationF1:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_pred: Tensor, y_true: Tensor):

        y_pred = y_pred.detach().sigmoid().cpu().view(-1)
        y_true = y_true.detach().cpu().view(-1)

        if self.threshold is not None:
            y_pred = y_pred > float(self.threshold)

        y_true = y_true.float()
        y_pred = y_pred.float()

        self.y_pred.extend(y_pred)
        self.y_true.extend(y_true)

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        if len(self.y_true) == 0:
            return 0
        return f1_score(self.y_true, self.y_pred)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix + '/value', self.value(), step)
