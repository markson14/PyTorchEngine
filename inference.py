# internal lib

from engine.utils import seed_reproducer
# external lib
import logging

import os
from engine.config import get_cfg, get_outname
from engine.utils.logger import setup_logger
from engine.engine import Tester

seed_reproducer(42)


def setup():
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    cfg.merge_from_file('./myconfig.yaml')
    output_name = get_outname(cfg)
    cfg.merge_from_list(['OUTPUT_NAME', output_name])
    cfg.merge_from_file(f'./log/{cfg.TASK}/{cfg.DATE}/{output_name}/myconfig.yaml')
    cfg.freeze()
    return cfg


def inference():
    config = setup()
    # output path
    logger = logging.getLogger()
    log_path = './log/{}/{}/{}'.format(config.TASK, config.DATE, config.OUTPUT_NAME)
    os.makedirs(log_path, exist_ok=True)
    inference_path = os.path.join(log_path, 'inference')
    if not os.path.exists(inference_path):
        os.mkdir(inference_path)
    setup_logger(config, inference_path, inference=True)
    logger.info(config)
    tester = Tester(logger=logger,
                    inference_path=inference_path,
                    cfg=config)
    tester.inference()


if __name__ == "__main__":
    inference()
