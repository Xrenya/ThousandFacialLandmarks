import torch
import torch.nn as nn



class Regressor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.head(x)
        return x
