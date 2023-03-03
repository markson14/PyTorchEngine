# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry
import torch.nn as nn

LOSS_REGISTRY = Registry("LOSS")  # noqa F401 isort:skip
LOSS_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_loss(cfg):
    """
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    """
    loss_name = cfg.SOLVER.LOSS.NAME
    loss = LOSS_REGISTRY.get(loss_name)
    return loss
