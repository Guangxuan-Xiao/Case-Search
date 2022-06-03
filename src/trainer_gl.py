import torch
import wandb
import time
import sys
from datetime import datetime


from icecream import ic

import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, loader, optimizer, loss_fn, scheduler, evaluator, device, logger, config, verbose=True):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.device = device
        self.stream = torch.cuda.Stream(self.device)
        self.device_id = int(config.device.split(':')[-1])
        self.logger = logger
        self.num_epochs = int(config.train.num_epochs)
        self.early_stop_patience = int(config.train.early_stop_patience)
        self.config = config
        self.verbose = verbose
        self.eval_interval = int(config.train.eval_interval)
        self.best_val_metric = -float('inf')
        self.best_test_metric = -float('inf')
        self.log_level = int(config.performance.log_level)
        self.sync = config.performance.sync
        self.do_test = config.train.num_test_samples != 0

    def loader_epoch(self):
        # Just iterate over the loader
        num_batches = num_nodes = num_edges = num_seeds = 0
        tbegin = time.time()
        for batch in tqdm(self.loader.train_loader):
            num_batches += 1
            y_true = batch.y
            num_nodes += batch.x.shape[0]
            num_edges += batch.idx.shape[0] if "CSR" in self.model.graph_type else batch.edge_index.shape[1]
            num_seeds += y_true.shape[0]
        tend = time.time()
        ret_dict = {'time': tend - tbegin,
                    '#nodes/B': num_nodes / num_batches, '#edges/B': num_edges / num_batches, '#seeds': num_seeds,
                    'throughput': num_nodes / (tend - tbegin), 'iter_time': (tend - tbegin) / num_batches, 'date': str(datetime.now()), "num_iter": num_batches}
        return ret_dict

    def train_epoch(self):
        self.model.train()
        train_loss = num_batches = 0
        y_preds, y_trues = [], []

        for batch in tqdm(self.loader.train_loader):
            num_batches += 1
            self.optimizer.zero_grad()
            output = self.model(batch)
            y_pred, y_true = output[batch.mask], batch.y
            loss = self.loss_fn(y_pred, y_true)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y_true.detach().cpu())
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        metric = self.evaluator(y_preds, y_trues).item()
        ret_dict = {'loss': train_loss / num_batches, 'metric': metric}
        return ret_dict

    @torch.no_grad()
    def evaluate_epoch(self, split='val'):
        self.model.eval()
        total_loss = num_batches = num_nodes = num_edges = 0
        y_preds, y_trues = [], []
        if split == 'val':
            loader = self.loader.val_loader
        elif split == 'test':
            loader = self.loader.test_loader
        else:
            raise ValueError('Invalid split: {}'.format(split))
        for batch in tqdm(loader):
            output = self.model(batch)
            y_pred, y_true = output[batch.mask], batch.y
            loss = self.loss_fn(y_pred, y_true)
            total_loss += loss.item()
            num_batches += 1
            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y_true.detach().cpu())
            num_nodes += batch.x.shape[0]
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        metric = self.evaluator(y_preds, y_trues)
        tend = time.time()
        ret_dict = {'loss': total_loss / num_batches, 'metric': metric.item()}
        return ret_dict

    def train(self):
        # nvmlInit()
        early_stop_counter = 0
        print(self.model)
        for self.epoch in range(1, self.num_epochs+1):
            train_dict = self.train_epoch()
            self.logger.write_epoch(
                {'train_'+k: v for k, v in train_dict.items()}, epoch=self.epoch)
            if self.epoch % self.eval_interval == 0:
                val_dict = self.evaluate_epoch(split='val')
                self.logger.write_epoch(
                    {'val_'+k: v for k, v in val_dict.items()}, epoch=self.epoch)
                if self.do_test:
                    test_dict = self.evaluate_epoch(split='test')
                    self.logger.write_epoch(
                        {'test_'+k: v for k, v in test_dict.items()}, epoch=self.epoch)
                if val_dict['metric'] > self.best_val_metric:
                    self.best_val_metric = val_dict['metric']
                    if self.do_test:
                        self.best_test_metric = test_dict['metric']
                    early_stop_counter = 0
                    self.logger.dump_model(self.model)
                else:
                    early_stop_counter += 1
                if early_stop_counter >= self.early_stop_patience:
                    break
        self.logger.write_final({
            "best_val_metric": self.best_val_metric,
            "best_test_metric": self.best_test_metric
        })
