import torch.nn as nn

from engine.config import configurable
from engine.loss.build import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    @configurable
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}
