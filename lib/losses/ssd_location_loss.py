from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SSDLocationLoss(_Loss):
    def __init__(self):
        super(SSDLocationLoss, self).__init__()

    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        '''Compute loss between (loc_preds, loc_targets)

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''

        pos = true_labels > 0  # [N,#anchors]
        mask = pos.unsqueeze(2).expand_as(pred_bboxes)  # [N,#anchors,4]

        predictions = pred_bboxes[mask].view(-1, 4)
        targets = true_bboxes[mask].view(-1, 4)

        num_pos = pos.sum().float()
        num_pos = torch.clamp(num_pos, min=1.)  # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes

        loc_loss = F.smooth_l1_loss(predictions, targets, reduction='sum')
        loc_loss = loc_loss / num_pos
        # print('SSDLocationLoss', float(loc_loss), int(num_pos))
        return loc_loss

