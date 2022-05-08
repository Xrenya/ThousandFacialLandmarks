from abc import ABC, abstractmethod
import torch.nn as nn
from torchvision.models import resnet18


class Head(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.head = None

    def forward(self, x, **kwargs):
        x = self.head(x)
        return x


class Regressor(Head):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.head = nn.Linear(in_features, out_features, bias=True)
