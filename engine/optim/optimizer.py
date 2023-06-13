from torch import optim
from .build import OPTIMIZER_REGISTRY
from engine.config import configurable


@OPTIMIZER_REGISTRY.register()
class SGD(optim.SGD):
    @configurable
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super(SGD, self).__init__(params,
                                  lr,
                                  momentum,
                                  dampening,
                                  weight_decay,
                                  nesterov,)

    @classmethod
    def from_config(cls, cfg, net_params):
        return {
            'params': net_params,
            'lr': cfg.SOLVER.OPTIMIZER.LR_START,
            'momentum': 0.9,
            'dampening': 0,
            'weight_decay': 0.99,
            'nesterov': False,
        }
