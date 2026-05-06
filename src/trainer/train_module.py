import logging
from typing import Dict, Iterable, Optional, Union

import numpy
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import get_data_loader
from evaluator.lipschitz_constant import bound_lipschitz_constant
from utils.line_formatter import LineFormatter
from utils.statistics import Statistics

from .lr_scheduler import PartialLRScheduler
from .metrics import Accuracy, Metric
from .optimizer import PartialOptimizer


class Train:
    eval_metrics: Dict[str, Union[Module, Metric]]
    metrics: Dict[str, numpy.ndarray]

    def __init__(self,
                 model: Module,
                 trainset: Dataset,
                 batch_size: int,
                 epochs: int,
                 loss: Module,
                 device: Union[str, torch.device],
                 optimizer: PartialOptimizer,
                 lr_scheduler: Optional[PartialLRScheduler] = None,
                 valset: Optional[Dataset] = None,
                 eval_metrics: Optional[Dict[str, Metric]] = None,
                 num_workers: int = 2,
                 logger=logging,
                 ):
        self.logger = logger
        self.epochs = epochs

        self.device = device
        self.model = model.to(self.device)

        self.input_shape = trainset[0][0].shape
        self.batch_size = batch_size
        self.train_loader, self.val_loader = get_data_loader(
            trainset, valset, batch_size, num_workers, self.logger.info)

        self.optimizer = optimizer(params=self.model.parameters())
        self.lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler is not None else None

        self.loss_fn = loss
        self.init_metrics(loss, eval_metrics)
        self.line_formatter = LineFormatter()

    def init_metrics(self, loss: Module, eval_metrics: Union[None, Dict[str, Metric]]):
        self.eval_metrics = dict(loss=loss)
        self.eval_metrics.update(dict(accuracy=Accuracy()))
        if eval_metrics is not None:
            self.eval_metrics.update(eval_metrics)

        metrics_aggs = dict(epoch='none')
        for name, metric in self.eval_metrics.items():
            metrics_aggs.update({'train_'+name: metric.aggregation})
            metrics_aggs.update({'val_'+name: metric.aggregation})

        self.metrics = dict(
            zip(metrics_aggs.keys(), [numpy.array([])]*len(metrics_aggs)))
        self.statistics = Statistics(metrics_aggs)
        self.logger.info('Statistics to be computed: ')
        self.logger.info(str(self.statistics))

    def update_weights(self, inputs: torch.Tensor, labels: torch.Tensor):
        out = self.model(inputs)
        loss = self.loss_fn(out, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach(), out.detach()

    @torch.no_grad()
    def evaluate_metrics(self, out: torch.Tensor, labels: torch.Tensor, split: str):
        for name, metric_fn in self.eval_metrics.items():
            self.update_metrics(**{split+'_'+name: tensor2numpy(metric_fn(out, labels))})

    def train_step(self, epoch: int):
        self.model.train()
        train_loop = tqdm(self.train_loader, desc=f'Train [{epoch}/{self.epochs}]')
        for (inputs, labels) in train_loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, out = self.update_weights(inputs, labels)
            self.evaluate_metrics(out, labels, 'train')
            train_loop.set_postfix(Loss=tensor2numpy(loss),
                                   Accuracy=self.metrics['train_accuracy'].mean())

    @torch.no_grad()
    @torch.nn.utils.parametrize.cached()
    def validation_step(self, epoch: int):
        self.model.eval()
        val_loop = tqdm(self.val_loader, desc=f'Validation[{epoch}/{self.epochs}]')
        for (inputs, labels) in val_loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.model(inputs)
            loss = self.loss_fn(out, labels)
            self.evaluate_metrics(out, labels, 'val')
            val_loop.set_postfix(Loss=tensor2numpy(loss),
                                 Accuracy=self.metrics['val_accuracy'].mean())

    def run(self, root_path: Optional[str] = None, save_freq: int = 10, save_state_dict=True, **addon_print):
        for epoch in range(1, 1+self.epochs):
            self.train_step(epoch)
            if self.val_loader is not None:
                self.validation_step(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.update_metrics(epoch=[epoch])
            self.statistics.update(**self.metrics)
            self.reset_metrics()

            self.print_stats(**addon_print)
            if root_path is not None and (epoch % save_freq == 0 or epoch == self.epochs):
                self.logger.info('Saving Checkpoint...')
                self.save(root_path, save_state_dict)
                self.logger.info('Done.')
        self.model.eval()
        ls_bound = bound_lipschitz_constant(
            self.model, self.input_shape[-1], 1000)
        self.logger.info(f'Lipschitz constant: {tensor2numpy(ls_bound)}')

    def update_metrics(self, **kwargs):
        for key, val in kwargs.items():
            self.metrics[key] = numpy.append(self.metrics[key], val)

    def reset_metrics(self):
        self.metrics = dict(
            zip(self.metrics.keys(), [numpy.array([])]*len(self.metrics)))

    def print_stats(self, **kwargs):
        logs = self.statistics.get_last()
        if self.lr_scheduler is not None:
            logs['learning_rate'] = self.lr_scheduler.get_last_lr()
        self.logger.info(self.line_formatter.create_line({**logs, **kwargs}))

    def save(self, path: str, save_state_dict: bool, **addon_save):
        if save_state_dict:
            self.model.eval()
            torch.save(self.model.state_dict(), path+'/model_state_dict.pth')
        self.statistics.save(path+'/training_statistics.csv')
        self.statistics.save_plot(path+'/training_statistics.png', subplots=True)


def tensor2numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
