import torch
from torch import Tensor

from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


def jaccard_loss(y_pred, y_true, from_logits=True, per_image=False, smooth=1e-4):
    batch_size = y_pred.size(0)

    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(batch_size, -1)
    y_true = y_true.view(batch_size, -1)

    if per_image:
        intersection = torch.sum(y_pred * y_true, dim=1)
        union = torch.sum(y_pred, dim=1) + torch.sum(y_true, dim=1) - intersection
    else:
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou


def focal_loss_with_logits(logits, target, gamma=2, alpha=0.25):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-logits * (target * 2 - 1))
    focal_term = alpha * (invprobs.exp() ** gamma)
    loss = focal_term * loss
    return loss.mean()


class ShipMaskLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred[:, 0, ...], y_true[:, 0, ...]
        # return focal_loss_with_logits(y_pred, y_true)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=5.)
        iou_loss = jaccard_loss(y_pred, y_true)
        return iou_loss * 0.5 + bce_loss


class ShipEdgeLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred[:, 1, ...], y_true[:, 1, ...]
        # return focal_loss_with_logits(y_pred, y_true)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=10.)
        iou_loss = jaccard_loss(y_pred, y_true)
        return iou_loss * 0.5 + bce_loss


if __name__ == '__main__':
    x = torch.rand((4, 2, 256, 256))
    x = x.cuda()

    y = torch.rand((4, 2, 256, 256))
    y = y.cuda()

    l1 = ShipMaskLoss()
    l2 = ShipEdgeLoss()
    print(l1(x, y))
    print(l2(x, y))
