import torch

from lib.common.torch_utils import count_parameters
from lib.models.encoders import SEResNeXt50Encoder, Resnet34Encoder, Resnet50Encoder
from lib.models.retinanet.rretina_net import RRetinaNetShared, RRetinaNet
from lib.models.rssd.rsdd import RSSD
from lib.models.ship_classifier_model import ShipClassifierModel
from lib.models.unet import UNetModel
from lib.modules.unet_decoder import UnetDecoderBlockSE, UnetCentralBlock


def get_model(model_name: str, num_classes=1, **kwargs):

    if model_name == 'rssd':
        return RSSD(num_classes=num_classes, **kwargs)

    if model_name == 'rretina34':
        return RRetinaNetShared(encoder=Resnet34Encoder(pretrained=True), num_classes=num_classes, **kwargs)

    if model_name == 'rretina50':
        return RRetinaNetShared(encoder=Resnet50Encoder(pretrained=True), num_classes=num_classes, **kwargs)

    if model_name == 'rretina_net':
        return RRetinaNetShared(encoder=SEResNeXt50Encoder(**kwargs), num_classes=num_classes, **kwargs)

    if model_name == 'rretina_net_v2':
        return RRetinaNet(encoder=SEResNeXt50Encoder(**kwargs), num_classes=num_classes, **kwargs)

    if model_name == 'seresnext_cls':
        return ShipClassifierModel(encoder=SEResNeXt50Encoder(**kwargs), num_classes=num_classes, **kwargs)

    raise ValueError(f'Unsupported model name {model_name}')


def test_se_resnext_unet():
    net = UNetModel(num_classes=1, encoder=SEResNeXt50Encoder, decoder_block=UnetDecoderBlockSE, central_block=UnetCentralBlock)
    net.eval()
    net.set_encoder_training_enabled(False)
    print(count_parameters(net))
    x = torch.rand((4, 3, 512, 512))

    mask = net(x)
    print(mask.size())
    print(net)

