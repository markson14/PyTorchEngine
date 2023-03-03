
import torch.nn as nn
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, RandomSampler
from engine.models import build_backbone, build_head, Model
from engine.loss import build_loss
from engine.optim.build import build_optimizer
from engine.utils import SmoothedValue, seed_reproducer
from engine.utils import seed_reproducer

seed_reproducer(42)


class Trainer:
    def __init__(
        self,
        logger=None,
        epoch=None,
        train_metric_logger=None,
        valid_metric_logger=None,
        cfg=None,
        setup_func=None,
    ) -> None:
        self.logger = logger
        self.epoch = epoch
        self.train_metric_logger = train_metric_logger
        self.valid_metric_logger = valid_metric_logger
        self.cfg = cfg
        self.setup_func = setup_func

        self.teacher_net = []

    def fit(self):
        self.valid_metric_logger.update(best=0)
        self.get_data()
        ep = self.get_model(self.cfg, teacher=False)
        self.get_parameters(self.net)
        # avoid warning message
        self.optimizer.zero_grad()
        self.optimizer.step()
        for epoch in range(ep, self.cfg.MAX_EPOCH+1):
            self.train(epoch)
            self.evaluate(epoch)
            self.save_checkpoint(epoch)
            torch.cuda.synchronize()

    def get_data(self):
        """get datasets and dataloaders"""
        # TODO: create your dataset class before you use it
        train_dataset = ...
        test_dataset = ...
        self.trainloader = DataLoader(train_dataset,
                                      batch_size=self.cfg.BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=self.cfg.NUM_WORKERS,
                                      pin_memory=True)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=self.cfg.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=self.cfg.NUM_WORKERS,
                                     pin_memory=True)
        self.logger.info(train_dataset)
        self.logger.info(test_dataset)

    def get_model(self, cfg):
        """get targeted model"""
        print('get {} model done...'.format(cfg.OUTPUT_NAME))
        # TODO: create your model & head
        backbone = build_backbone(cfg)
        head = build_head(cfg)
        self.net = Model(backbone, head)

    def get_parameters(self, net):
        """get loss and optimizer"""
        self.criterion = build_loss(cfg=self.cfg)
        self.optimizer = build_optimizer(cfg=self.cfg, net_params=net.parameters())

    def train(self, epoch):
        """Train one epoch"""
        # set net to train mode
        self.net.train()
        self.train_metric_logger.add_meter('lr', SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
        header = 'train epoch: [{}]'.format(epoch)
        for batch_idx, batch_input in enumerate(self.train_metric_logger.log_every(self.logger, self.trainloader, print_freq=100, header=header)):
            inputs, targets = self.cuda_input(batch_input)
            preds = self.net(inputs)
            # * loss compute
            loss = self.compute_loss(preds, targets)
            self.train_metric_logger.update(train_loss=loss)
            self.train_metric_logger.update(lr=self.optimizer.param_groups[0]['lr'])
            loss.backward()
            # optimizer.step
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, epoch):
        """Test one epoch"""
        # set net to eval mode
        self.net.eval()
        header = 'Valid epoch: [{}]'.format(epoch)
        for batch_idx, batch_input in enumerate(self.valid_metric_logger.log_every(self.logger, self.testloader, print_freq=100, header=header)):
            inputs, targets = self.cuda_input(batch_input)
            preds = self.net(inputs)

            # cal metrics
            eval_metric = {}
            # TODO: given preds & targets, calculate evaluation metrics. save in eval_metric

            self.valid_metric_logger.update(**eval_metric)

    def compute_loss(self, preds, targets):
        # TODO: given preds & targets, calculate loss
        return

    @staticmethod
    def cuda_input(inputs_tuple):
        inputs = inputs_tuple['inputs'].cuda()  # to cuda
        targets = inputs_tuple['targets'].cuda()
        return inputs, targets

    def save_checkpoint(self):
        # TODO: Define your save checkpoint function
        ...
