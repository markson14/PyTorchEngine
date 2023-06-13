from .config import CfgNode, configurable, get_outname


def get_cfg():
    from .defaults import _C
    return _C.clone()
