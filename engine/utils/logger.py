#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist


def setup_logger(config, logpth, inference=False):
    if inference:
        infer = '-inference'
    else:
        infer = ''
    logfile = '{}-{}{}.log'.format(config.OUTPUT_NAME,
                                   time.strftime('%Y-%m-%d-%H-%M-%S'),
                                   infer)
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
