"""
Custom logger.
Author: Johannes S. Fischer
GitHub: https://github.com/joh-fischer/logger
"""
import yaml
from datetime import datetime
import csv
import os
import torch


class Aggregator:
    def __init__(self):
        """
        Aggregate sum and average.
        """
        self.sum = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, log_dir: str = 'logs', name: str = None, include_time: bool = False):
        self.name = name if name else ''
        self.time = datetime.now().strftime('_%y-%m-%d_%H%M%S') if include_time else ''

        self.metrics = []
        self.hparams = {}

        self.running_epoch = -1

        self.epoch = {}

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def log_hparams(self, params: dict):
        """
        Log hyper-parameters.
        Parameters
        ----------
        params : dict
            Dictionary containing the hyperparameters as key-value pairs.
            For example {'optimizer': 'Adam', 'lr': 1e-02}.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: dict, step: int = None, phase: str = '', aggregate: bool = False, n: int = 1):
        """
        Log metrics.
        Parameters
        ----------
        metrics_dict : dict
            Dictionary containing the metrics as key-value pairs. For example {'acc': 0.9, 'loss': 0.2}.
        step : int, optional
            Step number where metrics are to be recorded.
        phase : str
            Current phase, e.g. 'train' or 'val' or 'epoch_end'.
        aggregate : bool
            If true, aggregates values into sum and average.
        n : int
            Count for aggregating values.
        """
        step = step if step is not None else len(self.metrics)

        metrics = {'epoch': self.running_epoch, 'phase': phase}
        metrics.update({k: self._handle_value(v) for k, v in metrics_dict.items()})
        metrics['step'] = step

        self.metrics.append(metrics)

        if aggregate:
            for metric_name, metric_val in metrics_dict.items():
                if metric_name not in self.epoch:
                    self.epoch[metric_name] = Aggregator()
                self.epoch[metric_name].update(metric_val, n)

    def init_epoch(self, epoch: int = None):
        """
        Initializes a new epoch.
        Parameters
        ----------
        epoch : int
            If empty, running epoch will be increased by 1.
        """
        if epoch:
            self.running_epoch = epoch
        else:
            self.running_epoch += 1

        self.epoch = {}

    def save(self):
        """
        Save the hyperparameters and metrics to a file.
        """
        if self.metrics:
            metrics_file_path = os.path.join(self.log_dir, 'metrics' + self.name + self.time + '.csv')
            last_m = {}
            for m in self.metrics:
                last_m.update(m)
            metrics_keys = list(last_m.keys())

            with open(metrics_file_path, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=metrics_keys)
                writer.writeheader()
                writer.writerows(self.metrics)

        if self.hparams:
            hparams_file_path = os.path.join(self.log_dir, 'hparams' + self.name + self.time + '.yaml')

            output = yaml.dump(self.hparams, Dumper=yaml.Dumper)
            with open(hparams_file_path, 'w') as file:
                file.write(output)

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        return