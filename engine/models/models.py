import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 backbone,
                 heads):
        super(Model, self).__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, x):
        features = self.backbone(x)
        outs = self.head(features)
        return outs
