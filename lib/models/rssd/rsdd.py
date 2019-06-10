import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from lib.common.box_util import meshgrid_numpy
from lib.common.rbox_util import rbbox_iou, clip_to_tan_range, rbbox_nms
from lib.common.torch_utils import to_numpy, count_parameters
from lib.dataset.common import ID_KEY, rle_encode
from lib.models.fpnssd512.fpn import FPN50
from lib.visualizations import visualize_rbbox

RSSD_ORIENTATIONS = (0, 45, 90, 135)
RSSD_SCALE_RATIOS = [1., pow(2., 1. / 3.), pow(2., 2. / 3.)]
RSSD_ASPECT_RATIOS = [1.5, 3, 5]
RSSD_NUM_ANCHORS = len(RSSD_ORIENTATIONS) * len(RSSD_SCALE_RATIOS) * len(RSSD_ASPECT_RATIOS)
RSSD_NUM_PARAMS = 5


class RSSDBoxCoder:
    def __init__(self, image_width, image_height):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.anchor_areas = [16 * 16,
                             32 * 32.,
                             64 * 64.,
                             96 * 96.,
                             128 * 128.,
                             256 * 256.,
                             512 * 512.]  # 1

        self.anchor_boxes = self._get_anchor_boxes(input_size=np.array([image_width, image_height], dtype=np.float32))

    def _get_anchor_wht(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 3].
        '''
        anchor_wht = []
        for anchor_scale in self.anchor_areas:
            for aspect_ratio in RSSD_ASPECT_RATIOS:
                for theta in RSSD_ORIENTATIONS:
                    h = math.sqrt(anchor_scale / aspect_ratio)
                    w = aspect_ratio * h
                    assert w > h
                    for sr in RSSD_SCALE_RATIOS:  # scale
                        anchor_w = w * sr
                        anchor_h = h * sr
                        anchor_wht.append([anchor_w, anchor_h, theta])
        num_fms = len(self.anchor_areas)
        return np.array(anchor_wht, dtype=np.float32).reshape(num_fms, -1, 3)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (list) model input size of (w,h).

        Returns:
          anchor_boxes: (tensor) anchor boxes for each feature map. Each of size [#anchors,5],
            where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        anchor_wht = self._get_anchor_wht()
        feature_map_sizes = [np.ceil(input_size / pow(2., i + 3)) for i in range(num_fms)]

        boxes = []
        for i in range(num_fms):
            fm_size = feature_map_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])

            xy = meshgrid_numpy(fm_w, fm_h).astype(np.float32) + 0.5
            xy = np.tile((xy * grid_size).reshape(fm_h, fm_w, 1, 2), (1, 1, RSSD_NUM_ANCHORS, 1))
            wht = np.tile(anchor_wht[i].reshape(1, 1, RSSD_NUM_ANCHORS, 3), (fm_h, fm_w, 1, 1))
            box = np.concatenate([xy, wht], 3)  # [x,y,w,h,t]
            boxes.extend(box.reshape(-1, 5))
        return np.array(boxes, dtype=np.float32)

    def encode(self, boxes, labels, return_anchors=False):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
          tt = tan(theta - anchor_theta)

        Args:
          boxes: (np.ndarray) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj,4].
          labels: (np.ndarray) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,5].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''

        def argmax(x):
            '''Find the max value index(row & col) of a 2D tensor.'''
            i = np.unravel_index(np.argmax(x), x.shape)
            return i

        if len(boxes) == 0:
            return self.anchor_boxes.copy(), np.zeros((len(self.anchor_boxes)), dtype=int)

        ious = rbbox_iou(self.anchor_boxes, boxes)  # [#anchors, #obj]

        index = np.zeros(len(self.anchor_boxes), dtype=np.long)
        index.fill(-1)

        masked_ious = ious.copy()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1) >= 0.4)
        if mask.any():
            index[mask] = np.argmax(ious[mask], 1)

        boxes = boxes[np.clip(index, a_min=0, a_max=None)]  # negative index not supported

        loc_targets = self.encode_boxes(boxes)
        cls_targets = 1 + labels[np.clip(index, a_min=0, a_max=None)]
        cls_targets[index < 0] = 0

        if return_anchors:
            return loc_targets, cls_targets, self.anchor_boxes[cls_targets > 0]

        return loc_targets, cls_targets

    def encode_boxes(self, boxes):
        xy_boxes = boxes[:, 0:2]
        wh_boxes = boxes[:, 2:4]
        tt_boxes = boxes[:, 4:5]

        xy_anchors = self.anchor_boxes[:, 0:2]
        wh_anchors = self.anchor_boxes[:, 2:4]
        tt_anchors = self.anchor_boxes[:, 4:5]

        eps = 1e-3

        loc_xy = (xy_boxes - xy_anchors) / wh_anchors
        loc_wh = np.log(np.clip(wh_boxes / wh_anchors, a_min=eps, a_max=None))
        loc_tt = tt_boxes - tt_anchors
        loc_tt = clip_to_tan_range(loc_tt)

        assert (-90 <= loc_tt).all()
        assert (loc_tt <= 90).all()

        loc_tt = np.clip(loc_tt, a_min=-90 + eps, a_max=90 - eps)
        loc_tt = np.deg2rad(loc_tt)
        loc_tt = np.tan(loc_tt)

        loc_targets = np.concatenate([loc_xy, loc_wh, loc_tt], axis=1)
        return loc_targets

    def decode_boxes(self, loc_preds):
        xy_boxes = loc_preds[:, 0:2]
        wh_boxes = loc_preds[:, 2:4]
        tt_boxes = loc_preds[:, 4:5]

        xy_anchors = self.anchor_boxes[:, 0:2]
        wh_anchors = self.anchor_boxes[:, 2:4]
        tt_anchors = self.anchor_boxes[:, 4:5]

        xy = xy_boxes * wh_anchors + xy_anchors

        # Clip bbox center to stay in range of image
        xy[:, 0] = np.clip(xy[:, 0], a_min=0, a_max=self.image_width - 1)
        xy[:, 1] = np.clip(xy[:, 1], a_min=0, a_max=self.image_height - 1)

        wh = np.exp(wh_boxes) * wh_anchors
        wh = np.clip(wh, a_min=1., a_max=None)  # Clip bbox size to min of 1 pixel

        tt = np.rad2deg(np.arctan(tt_boxes)) + tt_anchors
        # t = np.expand_dims(np.rad2deg(loc_preds[:, 4]) + tt_anchors, -1)

        boxes = np.concatenate([xy, wh, tt], 1)
        return boxes

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.2):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [#anchors,5].
          cls_preds: (tensor) predicted conf, sized [#anchors,#classes].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
          offsets: (tensor) offsets in global coordinate space for bounding boxes [#anchors,5].

        Returns:
          boxes: (tensor) rbbox locations, sized [#obj,5]. Format xywht
          labels: (tensor) class labels, sized [#obj,].
        '''
        box_preds = self.decode_boxes(loc_preds)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.shape[1]
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                # print('No candidates with confidence greater', score_thresh)
                continue

            # print('Candidates', np.sum(mask))

            box = box_preds[mask]
            score = score[mask]

            keep = rbbox_nms(box, score, nms_thresh)

            boxes.extend(box[keep])
            labels.extend(np.ones_like(keep) * i)
            scores.extend(score[keep])

        # print('Boxes', len(boxes))

        boxes = np.array(boxes)
        labels = np.array(labels)
        scores = np.array(scores)

        return boxes, labels, scores


class RSSD(nn.Module):
    num_loc_params = 5  # xc,yc,w,h,theta

    def __init__(self, num_classes, image_size=(512, 512), pretrained=True, **kwargs):
        super(RSSD, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes + 1  # Dummy class
        self.loc_head = self._make_head(RSSD_NUM_ANCHORS * RSSD_NUM_PARAMS)
        self.cls_head = self._make_head(RSSD_NUM_ANCHORS * self.num_classes)
        self.box_coder = RSSDBoxCoder(image_size[0], image_size[1])

        # See section "Inference and Training/Initialization" of https://arxiv.org/pdf/1708.02002.pdf
        pi = 0.01
        bias = - math.log((1 - pi) / pi)
        self.cls_head[-1].bias.data.zero_().add_(bias)

        resnet_state = resnet50(pretrained=pretrained).state_dict()
        self.fpn.load_state_dict(resnet_state, strict=False)

    def forward(self, image):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(image)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1,
                                                                      RSSD_NUM_PARAMS)  # [N, 9*NP,H,W] -> [N,H,W, 9*NP] -> [N,H*W*9, NP]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(image.size(0), -1,
                                                                      self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        bboxes = torch.cat(loc_preds, 1)
        labels = torch.cat(cls_preds, 1)

        return bboxes, labels

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def predict_as_csv(self, dataset, batch_size=1, workers=0):
        self.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

        image_ids = []
        rles = []

        for image, y in tqdm(dataloader, total=len(dataloader)):
            image = image.cuda(non_blocking=True)
            pred_boxes, pred_labels = self(image)
            for image_id, boxes, scores in zip(y[ID_KEY], to_numpy(pred_boxes), to_numpy(pred_labels)):
                boxes, _, scores = dataset.box_coder.decode(boxes, scores)
                if len(boxes):
                    mask = np.zeros((768, 768), dtype=np.uint8)

                    # First, we resterize all masks
                    for i, rbox in enumerate(boxes):
                        visualize_rbbox(mask, rbox, color=(i + 1, i + 1, i + 1), thickness=cv2.FILLED)

                    # Second, we do rle encoding. This prevents assigning same pixel to multiple instances
                    for i, rbox in enumerate(boxes):
                        rle = rle_encode(mask == (i + 1))
                        image_ids.append(image_id)
                        rles.append(rle)

                else:
                    image_ids.append(image_id)
                    rles.append(None)

        return pd.DataFrame.from_dict({'ImageId': image_ids, 'EncodedPixels': rles})


def test():
    model = RSSD(1)
    image = torch.randn(4, 3, 512, 512)
    output = model(image)
    print(count_parameters(model))

    model.cuda()
    image = image.cuda()

    with torch.cuda.profiler.profile() as prof:
        model(image)  # Warmup CUDA memory allocator and profiler
        with torch.autograd.profiler.emit_nvtx():
            model(image)

    print(prof)


if __name__ == '__main__':
    test()
