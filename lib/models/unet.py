import torch
from torch import nn
from torch.nn import functional as F

from lib.models.base_models import EncoderModule
from lib.modules.conv_bn_act import CABN


class UNetModel(nn.Module):
    def __init__(self, num_classes, encoder: EncoderModule, decoder_block: nn.Module, central_block: nn.Module, num_filters=64, **kwargs):
        super().__init__()
        self.encoder = encoder(**kwargs)

        n_encoder_filters = self.encoder.output_filters
        n_decoder_filters = [num_filters * (i + 1) for i in range(len(n_encoder_filters))]

        center_in = n_encoder_filters[-1]
        center_out = n_decoder_filters[-1]
        self.center = central_block(center_in, center_out, **kwargs)

        decoders = []
        for n_enc, n_dec_prev, n_dev in zip(n_encoder_filters, n_decoder_filters[1:] + [center_out], n_decoder_filters):
            # print(n_enc, n_dec_prev, n_dev)
            decoder = decoder_block(n_enc, n_dec_prev, n_dev)
            decoders.append(decoder)

        # decoder_block(in_features, out_features, **kwargs) for in_features, out_features in zip(self.encoder.)])
        self.decoder = nn.ModuleList(reversed(decoders))

        # If first encoder layer outputs feature map of reduced size, we upsample it with additional smoothing convolution before computing logits
        if self.encoder.output_strides[0] != 1:
            self.smooth = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, image):
        # Compute features for all encoder layers
        features = self.encoder(image)

        # Reverse them to make easier management
        features = list(reversed(features))
        n_features = len(features)

        x = self.center(features[0])

        for i in range(n_features):
            x = self.decoder[i](x, features[i])

        if self.encoder.output_strides[0] != 1:
            x = F.interpolate(x, size=image.size()[2:], mode='bilinear', align_corners=True)
            x = self.smooth(x)

        x = self.final(x)
        return x

    def set_encoder_training_enabled(self, enabled):
        for layer in self.encoder.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)
