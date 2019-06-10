import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules.abn import ABN, ACT_RELU
from lib.modules.conv_bn_act import CABN
from lib.modules.coord_conv import append_coords
from lib.modules.gate2d import ChannelSpatialGate2d


class UnetCentralBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, **kwargs):
        super().__init__()
        self.conv1 = CABN(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, **kwargs)
        self.conv2 = CABN(out_filters, out_filters, kernel_size=3, padding=1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoderBlock(nn.Module):
    """
    """

    def __init__(self, in_dec_filters, in_enc_filters, out_filters, abn_block=ABN, activation=ACT_RELU, pre_dropout_rate=0., post_dropout_rate=0., **kwargs):
        super(UnetDecoderBlock, self).__init__()

        self.conv1 = CABN(in_dec_filters + in_enc_filters, out_filters, kernel_size=3, stride=1, padding=1, abn_block=abn_block, activation=activation, **kwargs)
        self.conv2 = CABN(out_filters, out_filters, kernel_size=3, stride=1, padding=1, abn_block=abn_block, activation=activation, **kwargs)

        self.pre_drop = nn.Dropout(pre_dropout_rate, inplace=True)
        self.post_drop = nn.Dropout(post_dropout_rate, inplace=True)

    def forward(self, x, enc):
        lat_size = enc.size()[2:]
        x = F.interpolate(x, size=lat_size, mode='bilinear', align_corners=True)

        x = torch.cat([x, enc], 1)

        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.post_drop(x)
        return x


class UnetDecoderBlockSE(UnetDecoderBlock):
    """
    Decoder with Squeeze & Excitation block
    """

    def __init__(self, in_dec_filters, in_enc_filters, out_filters, abn_block=ABN, activation=ACT_RELU, up=True, pre_dropout_rate=0., post_dropout_rate=0., **kwargs):
        super().__init__(in_dec_filters, in_enc_filters, out_filters, abn_block, activation, pre_dropout_rate, post_dropout_rate, **kwargs)
        self.scse = ChannelSpatialGate2d(out_filters)

    def forward(self, x, enc=None):
        x = super().forward(x, enc)
        x = self.scse(x)
        return x


class UnetDecoderBlockSECoord(UnetDecoderBlockSE):
    """
    Decoder with Squeeze & Excitation block and CoordConv
    """

    def __init__(self, in_dec_filters, in_enc_filters, out_filters, abn_block=ABN, activation=ACT_RELU, up=True, pre_dropout_rate=0., post_dropout_rate=0., **kwargs):
        super().__init__(in_dec_filters + 2, in_enc_filters, out_filters, abn_block, activation, up, pre_dropout_rate, post_dropout_rate, **kwargs)

    def forward(self, x, enc=None):
        x_coord = append_coords(x)
        x = super().forward(x_coord, enc)
        return x
