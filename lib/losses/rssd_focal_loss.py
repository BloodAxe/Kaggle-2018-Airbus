from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from lib.losses.functional import binary_focal_loss_with_logits, binary_focal_loss


class RSSDFocalLoss(_Loss):
    def __init__(self):
        super(RSSDFocalLoss, self).__init__()


    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          pred_bboxes: (tensor,float) predicted locations, sized [batch_size, #anchors, 5].
          pred_labels: (tensor,float) encoded target locations, sized [batch_size, #anchors, #classes].
          true_bboxes: (tensor,float) predicted class confidences, sized [batch_size, #anchors, 5].
          true_labels: (tensor,long) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        if len(true_labels.size()) != 2:
            raise ValueError('Shape of the true_labels must be [batch_size, #anchors]')

        if len(pred_labels.size()) != 3:
            raise ValueError('Shape of the true_labels must be [batch_size, #anchors, #classes]')

        assert pred_bboxes.size(0) == true_bboxes.size(0)
        assert pred_labels.size(0) == true_labels.size(0)

        # for in zip(pred_bboxes, pred_labels, true_bboxes, true_labels):

        pos = true_labels > 0  # [N,#anchors]
        true_labels = true_labels.float()

        num_pos = torch.clamp(pos.sum().float(), min=1.)  # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes

        pos_neg = true_labels > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(pred_labels)
        masked_cls_preds = pred_labels[mask].view(-1, 1)
        cls_loss = binary_focal_loss(masked_cls_preds, true_labels[pos_neg], alpha=0.05, gamma=2)

        loss = cls_loss.mean()
        # loss = cls_loss.sum() / num_pos
        return loss
