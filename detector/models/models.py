from abc import ABC, abstractmethod
import torch.nn as nn
from torchvision.models import resnet18, densenet169


class Backbone(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.backbone = None

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        return x


class ResNet18(Backbone):
    def __init__(self, pretrained):
        super().__init__()
        model = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*(list(model.children())))[:-1]


class DenseNet(Backbone):
    def __init__(self, pretrained):
        super().__init__()
        model = densenet169(pretrained=pretrained)
        self.backbone = nn.Sequential(*(list(model.children())))[:-1]

