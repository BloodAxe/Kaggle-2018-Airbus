from torch import nn

from lib.models.base_models import EncoderModule
from lib.modules.classification import ClassificationModule, MultiheadClassificationModule


class ShipClassifierModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes=1, **kwargs):
        super().__init__()
        self.encoder = encoder
        # self.classifier = ClassificationModule(self.encoder.output_filters[-1], num_classes)
        self.classifier = MultiheadClassificationModule(self.encoder.output_filters, num_classes)

    def forward(self, image):
        features = self.encoder(image)
        # last_features = features[-1]
        return self.classifier(features)
