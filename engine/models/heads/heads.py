import torch
import torch.nn as nn
from ..build import HEAD_REGISTRY
from ...config import configurable


@ HEAD_REGISTRY.register()
class Head(nn.Module):
    @ configurable
    def __init__(self):
        # TODO: create your head layers
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        # TODO: define your head forward steps
        pass

    @ classmethod
    def from_config(cls, cfg):
        return {}
