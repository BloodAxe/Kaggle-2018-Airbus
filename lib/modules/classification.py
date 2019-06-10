import torch
from torch import nn

from lib.modules.global_pooling import GlobalAvgPool2d, GlobalMaxPool2d


class ClassificationModule(nn.Module):
    def __init__(self, features, num_classes=1, dropout=0.25):
        super().__init__()
        self.maxpool = GlobalMaxPool2d()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiheadClassificationModule(nn.Module):
    def __init__(self, features, num_classes=1, reduction_rate=4, dropout=0.25):
        super().__init__()
        self.bottlenecks = nn.ModuleList(
            [nn.Conv2d(features[i], features[i] // reduction_rate, kernel_size=1, padding=0) for i in
             range(len(features))])
        self.maxpool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout, inplace=True)
        num_features = sum(f // reduction_rate for f in features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, features):
        pools = []
        for feature, bottleneck in zip(features, self.bottlenecks):
            x = self.maxpool(feature)
            x = self.dropout(x)
            x = bottleneck(x)
            pools.append(x)

        x = torch.cat(pools, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
