from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SSDLabelLoss(_Loss):
    # ===============================================================
    # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
    # ===============================================================
    def __init__(self):
        super(SSDLabelLoss, self).__init__()

    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        # cls_targets = expected[SSD_LABELS_KEY]
        # cls_preds = actual[SSD_LABELS_KEY]

        pos = true_labels > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_classes = pred_labels.size(2)

        cls_loss = F.cross_entropy(pred_labels.view(-1, num_classes),
                                   true_labels.view(-1), reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[true_labels < 0] = 0  # set ignored loss to 0

        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos | neg]

        num_pos = pos.sum().float()
        num_pos = torch.clamp(num_pos, min=1.)  # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes

        cls_loss = cls_loss.sum()
        # print('SSDLabelLoss', float(cls_loss), int(num_pos))
        return cls_loss / num_pos

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)  # [N,#anchors]

        num_neg = 3 * pos.sum(1)  # [N,]
        neg = rank < num_neg[:, None]  # [N,#anchors]
        return neg


class SSDBinaryLabelLoss(_Loss):
    """
    Label loss for binary targets
    """
    def __init__(self):
        super(SSDBinaryLabelLoss, self).__init__()

    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        true_labels = true_labels.float()

        pos = true_labels > 0  # [N,#anchors]
        batch_size = true_labels.size(0)

        cls_loss = F.binary_cross_entropy(pred_labels.view(-1), true_labels.view(-1), reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[true_labels < 0] = 0  # set ignored loss to 0

        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos | neg]

        # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes
        num_pos = torch.clamp(pos.sum().float(), min=1.)

        cls_loss = cls_loss.sum()
        return cls_loss / num_pos

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        a, idx = cls_loss.sort(1)  # sort by negative losses
        b, rank = idx.sort(1)  # [N,#anchors]

        num_neg = 3 * pos.sum(1)  # [N,]
        neg = rank < num_neg[:, None]  # [N,#anchors]
        return neg
