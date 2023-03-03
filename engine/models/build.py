# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry


BACKBONE_REGISTRY = Registry("BACKBONE")  # noqa F401 isort:skip
BACKBONE_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

HEAD_REGISTRY = Registry("HEAD")  # noqa F401 isort:skip
HEAD_REGISTRY.__doc__ = """
Registry for head.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_backbone(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.BACKBONE``.
    Note that it does not load any weights from ``cfg``.
    """
    backbone_name = cfg.MODEL.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)
    return backbone


def build_head(cfg):
    """
    Build the whole model head, defined by ``cfg.MODEL.HEAD``.
    Note that it does not load any weights from ``cfg``.
    """
    head_name = cfg.MODEL.HEAD.NAME
    head = HEAD_REGISTRY.get(head_name)(cfg)
    return head
