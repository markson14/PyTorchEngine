# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
OPTIMIZER_REGISTRY.__doc__ = """
Registry for optimizer, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_optimizer(cfg, net_params):
    """
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    """
    optimizer_name = cfg.SOLVER.OPTIMIZER.NAME
    optim = OPTIMIZER_REGISTRY.get(optimizer_name)(cfg, net_params)
    return optim
