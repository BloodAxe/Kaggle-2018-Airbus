from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class RSSDLocationLoss(_Loss):
    def __init__(self):
        super(RSSDLocationLoss, self).__init__()

    def forward(self, pred_bboxes, pred_labels, true_bboxes, true_labels):
        '''Compute loss between (loc_preds, loc_targets)

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 5].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 5].
        '''
        if pred_bboxes.size() != true_bboxes.size():
            raise RuntimeError(f"Shape of bbox tensors does not match {pred_bboxes.size()} {true_bboxes.size()}")

        pos = true_labels > 0  # [N,#anchors]
        mask = pos.unsqueeze(2).expand_as(pred_bboxes)  # [N,#anchors,5]

        if mask.size() != true_bboxes.size():
            raise RuntimeError(f"Shape of mask tensors does not match true_bboxes {mask.size()} {true_bboxes.size()}")

        if mask.size() != pred_bboxes.size():
            raise RuntimeError(f"Shape of mask tensors does not match pred_bboxes {mask.size()} {pred_bboxes.size()}")

        predictions = pred_bboxes[mask].view(-1, pred_bboxes.size(2))
        targets = true_bboxes[mask].view(-1, true_bboxes.size(2))

        # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes
        num_pos = torch.clamp(pos.sum().float(), min=1.)

        loc_loss = F.smooth_l1_loss(predictions, targets, reduction='sum')
        # xy_loss = F.smooth_l1_loss(predictions[:, 0:2], targets[:, 0:2], reduction='sum')
        # wh_loss = F.smooth_l1_loss(predictions[:, 2:4], targets[:, 2:4], reduction='sum')
        # tt_loss = F.smooth_l1_loss(predictions[:, 4:5], targets[:, 4:5], reduction='sum')
        # print('RSSDLocationLoss', float(xy_loss), float(wh_loss), float(tt_loss), int(num_pos))
        # loc_loss = (xy_loss + wh_loss + tt_loss)

        return loc_loss / num_pos
