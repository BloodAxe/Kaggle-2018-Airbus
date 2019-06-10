from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet50, resnet34

from lib.common.torch_utils import count_parameters
from lib.models.base_models import EncoderModule
from lib.models.senet_unet.senet import se_resnext50_32x4d


class SEResNeXt50Encoder(EncoderModule):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()

        encoder = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)
        self.layer0 = encoder.layer0
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return [4, 4, 8, 16, 32]

    @property
    def output_filters(self):
        return [64, 256, 512, 1024, 2048]



class Resnet34Encoder(EncoderModule):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()

        encoder = resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', encoder.conv1),
            ('bn1', encoder.bn1),
            ('relu', encoder.relu),
            ('maxpool', encoder.maxpool)
        ]))

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return [4, 4, 8, 16, 32]

    @property
    def output_filters(self):
        return [64, 64, 128, 256, 512]

    def forward(self, x):
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return features


class Resnet50Encoder(EncoderModule):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()

        encoder = resnet50(pretrained=pretrained, **kwargs)

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', encoder.conv1),
            ('bn1', encoder.bn1),
            ('relu', encoder.relu),
            ('maxpool', encoder.maxpool)
        ]))

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return [4, 4, 8, 16, 32]

    @property
    def output_filters(self):
        return [64, 256, 512, 1024, 2048]

    def forward(self, x):
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        return features


def test_SEResNeXt():
    net = SEResNeXt50Encoder().eval()
    print(count_parameters(net))
    x = torch.rand((4, 3, 512, 512))
    y = net(x)
    for yi in y:
        print(yi.size())


def test_Resnet50Encoder():
    net = Resnet50Encoder().eval()
    print(count_parameters(net))
    x = torch.rand((4, 3, 512, 512))
    y = net(x)
    for yi in y:
        print(yi.size())
