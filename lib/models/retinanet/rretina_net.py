import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.common.box_util import meshgrid_numpy
from lib.common.rbox_util import rbbox_iou, clip_to_tan_range, rbbox_nms
from lib.common.torch_utils import count_parameters, logit, to_numpy
from lib.losses.rssd_focal_loss import RSSDFocalLoss
from lib.losses.rssd_location_loss import RSSDLocationLoss
from lib.losses.ssd_label_loss import SSDBinaryLabelLoss
from lib.models.base_models import BaseDetectorModel, EncoderModule
from lib.models.encoders import Resnet50Encoder, SEResNeXt50Encoder, Resnet34Encoder
from lib.modules.conv_bn_act import CABN
from lib.modules.coord_conv import append_coords
from lib.modules.unet_decoder import UnetCentralBlock

SERSSD_ORIENTATIONS = (0, 45, 90, 135)
SERSSD_SCALE_RATIOS = [1., ]
SERSSD_ASPECT_RATIOS = [4.]
SERSSD_NUM_ANCHORS = len(SERSSD_ORIENTATIONS) * len(SERSSD_SCALE_RATIOS) * len(SERSSD_ASPECT_RATIOS)
SERSSD_NUM_PARAMS = 5


class RRNBoxCoder:
    """
    Box coder for Rotated Retina Net
    """

    def __init__(self, image_width, image_height, output_stride=[4, 8, 16, 32, 64]):
        self.input_size = np.array([image_width, image_height])
        self.output_stride = np.array(output_stride)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.feature_map_sizes = [np.ceil(self.input_size / os) for os in self.output_stride]

        self.anchor_areas = [
            8. * 8.,
            16 * 16,
            32 * 32.,
            64 * 64.,
            128 * 128.,
            # 256 * 256.,
            # 512 * 512.
        ]  # 1

        self.anchor_boxes = self._get_anchor_boxes(input_size=np.array([image_width, image_height], dtype=np.float32))

    def _get_anchor_wht(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 3].
        '''
        anchor_wht = []
        for anchor_scale in self.anchor_areas:
            for aspect_ratio in SERSSD_ASPECT_RATIOS:
                for theta in SERSSD_ORIENTATIONS:
                    h = math.sqrt(anchor_scale / aspect_ratio)
                    w = aspect_ratio * h
                    assert w > h
                    for sr in SERSSD_SCALE_RATIOS:  # scale
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

        boxes = []
        for i in range(num_fms):
            fm_size = self.feature_map_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])

            xy = meshgrid_numpy(fm_w, fm_h).astype(np.float32) + 0.5
            xy = np.tile((xy * grid_size).reshape(fm_h, fm_w, 1, 2), (1, 1, SERSSD_NUM_ANCHORS, 1))
            wht = np.tile(anchor_wht[i].reshape(1, 1, SERSSD_NUM_ANCHORS, 3), (fm_h, fm_w, 1, 1))
            box = np.concatenate([xy, wht], 3)  # [x,y,w,h,t]
            box = box.reshape(-1, 5)
            # print("Feature map", i, box.shape)
            boxes.extend(box)
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

        '''
        if len(boxes) == 0:
            loc_targets = self.encode_boxes(self.anchor_boxes)
            cls_targets = np.zeros(len(self.anchor_boxes), dtype=int)
        else:
            boxes = to_numpy(boxes)
            ious = rbbox_iou(self.anchor_boxes, boxes)  # [#anchors, #obj]

            max_ious_1, max_ids_1 = torch.from_numpy(ious).max(1)
            max_ious_0, max_ids_0 = torch.from_numpy(ious).max(0)
            max_ids = max_ids_1.numpy()
            max_ids[max_ids_0.numpy()] = np.arange(len(boxes))
            boxes = boxes[max_ids]
            loc_targets = self.encode_boxes(boxes)
            max_ious = max_ious_1.numpy()

            cls_targets = np.ones(len(self.anchor_boxes), dtype=int) * -1
            cls_targets[max_ious < 0.4] = 0
            cls_targets[max_ious >= 0.5] = 1
            cls_targets[max_ids_0] = 1

        if return_anchors:
            return loc_targets, cls_targets, self.anchor_boxes[cls_targets > 0]

        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.5, nms_thresh=0.5):
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

        loc_preds = to_numpy(loc_preds)
        cls_preds = to_numpy(cls_preds)

        boxes = self.decode_boxes(loc_preds)

        mask = cls_preds > score_thresh

        boxes = boxes[mask]
        scores = cls_preds[mask]

        keep = rbbox_nms(boxes, scores, threshold=nms_thresh)
        good_boxes = boxes[keep]
        good_scores = scores[keep]
        return good_boxes, good_scores

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


class RRNClassHead(nn.Module):
    def __init__(self, nin, num_classes, num_anchors=SERSSD_NUM_ANCHORS, pi=0.01):
        """

        :param nin:
        :param num_classes: Number of classes
        :param num_anchors: Number of anchors per pixel
        :param pi: Prior probability of the target class
        """
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.final = nn.Conv2d(nin, self.num_classes * num_anchors, kernel_size=3, padding=1)

        # See section "Inference and Training/Initialization" of https://arxiv.org/pdf/1708.02002.pdf
        # bias = - math.log((1 - pi) / pi)
        # self.final.bias.data.zero_().add_(bias)

    def forward(self, x):
        bs = x.size(0)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        cls_pred = self.final(x).sigmoid()

        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)  # [N,9*NC,H,W] -> [N,H,W,9*NC] -> [N,H*W*9,NC]
        return cls_pred


class RRNLocationHead(nn.Module):
    def __init__(self, nin, num_anchors=SERSSD_NUM_ANCHORS):
        super().__init__()
        self.conv1 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(nin, nin, kernel_size=3, padding=1)
        self.final = nn.Conv2d(nin, SERSSD_NUM_PARAMS * num_anchors, kernel_size=3, padding=1)

    def forward(self, x):
        bs = x.size(0)
        # x = append_coords(x)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        loc_pred = self.final(x)
        loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, SERSSD_NUM_PARAMS)  # [N, 9*NP,H,W] -> [N,H,W, 9*NP] -> [N,H*W*9, NP]
        return loc_pred


class RRNDecoderBlock(nn.Module):
    def __init__(self, in_enc_filters, in_dec_filters, out_filters):
        super().__init__()
        self.enc_conv = nn.Conv2d(in_enc_filters, out_filters, kernel_size=3, padding=1)
        #self.dec_conv = CABN(in_dec_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x, enc):
        enc = self.enc_conv(enc)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x + enc


class RRetinaNet(BaseDetectorModel):
    def __init__(self, encoder: EncoderModule, num_classes, image_size=(512, 512), pretrained=True, filters=128, **kwargs):
        super(RRetinaNet, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.box_coder = RRNBoxCoder(image_size[0], image_size[1])

        n_encoder_filters = self.encoder.output_filters
        n_decoder_filters = [filters for i in range(len(n_encoder_filters))]

        center_in = n_encoder_filters[-1]
        center_out = n_decoder_filters[-1]
        self.center = UnetCentralBlock(center_in, center_out)

        decoders = []
        cls_heads = []
        loc_heads = []

        self.center_loc_head = RRNLocationHead(center_out)
        self.center_cls_head = RRNClassHead(center_out, self.num_classes)

        for i in reversed(range(1, len(n_encoder_filters))):
            decoders.append(RRNDecoderBlock(n_encoder_filters[i], n_decoder_filters[i - 1], n_decoder_filters[i]))
            cls_heads.append(RRNClassHead(n_decoder_filters[i], self.num_classes))
            loc_heads.append(RRNLocationHead(n_decoder_filters[i]))

        self.decoders = nn.ModuleList(decoders)
        self.loc_heads = nn.ModuleList(loc_heads)
        self.cls_heads = nn.ModuleList(cls_heads)

    def forward(self, image):
        # Compute features for all encoder layers
        features = self.encoder(image)

        n_features = len(features)

        x = self.center(features[-1])

        loc_preds = [self.center_loc_head(x)]
        cls_preds = [self.center_cls_head(x)]

        for i, decoder, cls_head, loc_head in zip(reversed(range(1, n_features)), self.decoders, self.cls_heads, self.loc_heads):
            x = decoder(x, features[i])

            # For last (finest) layer, we scale up to have output stride of 2
            # if i == n_features - 1:
            #     x = F.interpolate(x, scale_factor=2, align_corners=True, mode='bilinear')
            loc_pred = loc_head(x)
            cls_pred = cls_head(x)

            # print('loc_pred[i]', i, loc_pred.size(), cls_pred.size())

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        loc_preds = list(reversed(loc_preds))
        cls_preds = list(reversed(cls_preds))

        bboxes = torch.cat(loc_preds, 1)
        labels = torch.cat(cls_preds, 1)
        # print('bboxes', bboxes.size())
        return bboxes, labels

    def set_encoder_training_enabled(self, enabled):
        for layer in self.encoder.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)


class RRetinaNetShared(BaseDetectorModel):
    def __init__(self, encoder: EncoderModule, num_classes, image_size=(512, 512), filters=256, **kwargs):
        super(RRetinaNetShared, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.box_coder = RRNBoxCoder(image_size[0], image_size[1])

        n_encoder_filters = self.encoder.output_filters
        n_decoder_filters = [filters for i in range(len(n_encoder_filters))]

        center_in = n_encoder_filters[-1]
        center_out = n_decoder_filters[-1]
        self.center = UnetCentralBlock(center_in, center_out)

        decoders = []

        for i in reversed(range(1, len(n_encoder_filters))):
            decoder = RRNDecoderBlock(n_encoder_filters[i], n_decoder_filters[i - 1], n_decoder_filters[i])
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.loc_head = RRNLocationHead(filters)
        self.cls_head = RRNClassHead(filters, num_classes)

    def forward(self, image):
        # Compute features for all encoder layers
        features = self.encoder(image)

        n_features = len(features)

        x = self.center(features[-1])

        loc_preds = [self.loc_head(x)]
        cls_preds = [self.cls_head(x)]

        for i, decoder in zip(reversed(range(1, n_features)), self.decoders):
            x = decoder(x, features[i])

            # For last (finest) layer, we scale up to have output stride of 2
            # if i == n_features - 1:
            #     x = F.interpolate(x, scale_factor=2, align_corners=True, mode='bilinear')
            loc_pred = self.loc_head(x)
            cls_pred = self.cls_head(x)

            # print('loc_pred[i]', i, loc_pred.size(), cls_pred.size())

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        loc_preds = list(reversed(loc_preds))
        cls_preds = list(reversed(cls_preds))

        bboxes = torch.cat(loc_preds, 1)
        labels = torch.cat(cls_preds, 1)
        # print('bboxes', bboxes.size())
        return bboxes, labels

    def set_encoder_training_enabled(self, enabled):
        for layer in self.encoder.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)


def test_loss():
    box_coder = RRNBoxCoder(512, 512)

    boxes = np.array([[40, 50, 100, 20, 15], [400, 200, 16, 4, -33]], dtype=np.float32)
    labels = np.array([0, 0], dtype=int)

    true_loc, true_cls = box_coder.encode(boxes, labels)
    focal_loss = RSSDFocalLoss(num_classes=1)
    location_loss = RSSDLocationLoss()

    true_loc = torch.from_numpy(true_loc)
    true_cls = torch.from_numpy(true_cls).long()

    true_loc = torch.unsqueeze(true_loc, 0)
    true_cls = torch.unsqueeze(true_cls, 0)

    floss = focal_loss(true_loc, torch.unsqueeze(logit(true_cls), -1), true_loc, true_cls)
    lloss = location_loss(true_loc, torch.unsqueeze(logit(true_cls), -1), true_loc, true_cls)
    print('Same input', floss, lloss)


@torch.no_grad()
def test_retinanet():
    print('RRetinaNet')
    model = RRetinaNet(encoder=SEResNeXt50Encoder(), num_classes=1, image_size=(768, 768))
    model.eval()
    model.set_encoder_training_enabled(False)

    boxes = np.array([[40, 50, 100, 20, 15], [200, 200, 16, 4, -33]], dtype=np.float32)
    labels = np.array([0, 0], dtype=int)
    true_loc, true_cls = model.box_coder.encode(boxes, labels)

    image = torch.randn(1, 3, model.box_coder.image_height, model.box_coder.image_width)
    pred_bboxes, pred_labels = model(image)
    print(count_parameters(model))
    print(pred_bboxes.size())
    print(pred_labels.size())

    focal_loss = RSSDFocalLoss()
    location_loss = RSSDLocationLoss()

    # Make tensor of proper shape & batch_size
    true_loc = torch.unsqueeze(torch.from_numpy(true_loc), 0)
    true_cls = torch.unsqueeze(torch.from_numpy(true_cls).long(), 0)

    floss = focal_loss(pred_bboxes, pred_labels, true_loc, true_cls)
    lloss = location_loss(pred_bboxes, pred_labels, true_loc, true_cls)
    print('Random input', floss, lloss)



@torch.no_grad()
def test_retinanet34_shared():
    model = RRetinaNetShared(encoder=Resnet34Encoder(), num_classes=1, image_size=(768, 768))

    model.set_encoder_training_enabled(False)
    model.eval()
    model.cuda()

    print(count_parameters(model))

    boxes = np.array([[40, 50, 100, 20, 15], [200, 200, 16, 4, -33]], dtype=np.float32)
    labels = np.array([0, 0], dtype=int)
    true_loc, true_cls = model.box_coder.encode(boxes, labels)

    image = torch.randn(1, 3, model.box_coder.image_height, model.box_coder.image_width)
    pred_bboxes, pred_labels = model(image.cuda())
    print(pred_bboxes.size())
    print(pred_labels.size())

    cls_loss = SSDBinaryLabelLoss()
    loc_loss = RSSDLocationLoss()

    # Make tensor of proper shape & batch_size
    true_loc = torch.unsqueeze(torch.from_numpy(true_loc), 0)
    true_cls = torch.unsqueeze(torch.from_numpy(true_cls).long(), 0)

    floss = cls_loss(pred_bboxes, pred_labels, true_loc.cuda(), true_cls.cuda())
    lloss = loc_loss(pred_bboxes, pred_labels, true_loc.cuda(), true_cls.cuda())
    print('Random input', floss, lloss)


