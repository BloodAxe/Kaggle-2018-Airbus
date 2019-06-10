import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn

from lib.common.torch_utils import count_parameters
from lib.models.base_models import BaseSegmentationModel
from lib.models.senet_unet.senet import senet154
from lib.modules.conv_bn_act import CABN
from lib.modules.unet_decoder import UnetDecoderBlock


class SenetUnet(BaseSegmentationModel):
    """
    -m senet_unet
    Batch size:
        Patch 256x256 - 8 images
    """
    def __init__(self, num_classes: int, pretrained=True):
        super().__init__()

        encoder = senet154(pretrained='imagenet' if pretrained else None)
        decoder = UnetDecoderBlock

        self.encoders = nn.ModuleList([encoder.layer0,
                                       encoder.layer1,
                                       encoder.layer2,
                                       encoder.layer3,
                                       encoder.layer4])

        self.center = nn.Sequential(
            CABN(2048, 1024, kernel_size=3, padding=1),
            CABN(1024, 512, kernel_size=3, padding=1),
        )

        dec_out = [512, 256, 128, 64, 64]

        self.decoder = nn.ModuleList([decoder(2048 + 512, dec_out[0], middle_channels=dec_out[0]),
                                      decoder(1024 + dec_out[0], dec_out[1], middle_channels=dec_out[1]),
                                      decoder(512 + dec_out[1], dec_out[2], middle_channels=dec_out[2]),
                                      decoder(256 + dec_out[2], dec_out[3], middle_channels=dec_out[3]),
                                      decoder(128 + dec_out[3], dec_out[4], middle_channels=dec_out[4], up=False)])

        self.final = nn.Sequential(
            CABN(dec_out[4], dec_out[4], kernel_size=3, padding=1),
            CABN(dec_out[4], dec_out[4], kernel_size=3, padding=1),
            nn.Conv2d(dec_out[4], num_classes, kernel_size=1))

    def get_encoder_features(self, image):
        features = []
        x = image
        for mod in self.encoders:
            x = mod(x)
            features.append(x)
        return features

    def forward(self, image):
        features = self.get_encoder_features(image)

        x = self.center(F.max_pool2d(features[-1], kernel_size=2, stride=2))

        for encoder_features, decoder_block in zip(reversed(features), self.decoder):
            x = decoder_block(x, encoder_features)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.final(x)
        return x


if __name__ == '__main__':
    net = SenetUnet(num_classes=2).eval()
    print(count_parameters(net))

    mask = net(torch.rand((4, 3, 256, 256)))
    print(mask.size())
