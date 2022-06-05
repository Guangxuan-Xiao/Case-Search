import torch
import wandb
import os.path as osp
from icecream import ic
from tqdm import tqdm
from evaluator import ndcg
import numpy as np


class Trainer:
    def __init__(self, model, loaders, optimizer, scheduler, device, logger, loss_fn, config):
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.epoch = 0
        self.best_val_metric = -float('inf')
        self.best_epoch_dict = None

    def train_epoch(self):
        self.model.train()
        train_loss, num_batches = 0, 0
        pbar = tqdm(self.loaders[self.config.mode])
        for batch in pbar:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            scores = self.model(batch.query_inputs,
                                batch.candidate_inputs)
            labels = batch.labels
            loss = self.loss_fn(scores.reshape(-1, 1), labels)
            train_loss += loss.item()
            pbar.set_description(
                f"Epoch {self.epoch} Train Loss: {loss.item():.4f}")
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            num_batches += 1
        return train_loss / num_batches

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        val_loss, num_batches = 0, 0
        total_scores, total_labels, total_query_ridxs = [], [], []
        pbar = tqdm(self.loaders['val'])
        for batch in pbar:
            batch = batch.to(self.device)
            scores = self.model(batch.query_inputs,
                                batch.candidate_inputs)
            labels = batch.labels
            loss = self.loss_fn(scores.reshape(-1, 1), labels)
            val_loss += loss.item()
            pbar.set_description(
                f"Epoch {self.epoch} Val Loss: {loss.item():.4f}")
            num_batches += 1
            total_scores.append(scores.flatten().detach().cpu())
            total_labels.append(labels.flatten().detach().cpu())
            total_query_ridxs.append(
                batch.query_ridxs.flatten().detach().cpu())

        total_scores = torch.cat(total_scores)
        total_labels = torch.cat(total_labels)
        total_query_ridxs = torch.cat(total_query_ridxs)

        query_ridxs, query_ridxs_idx = torch.unique(
            total_query_ridxs, return_inverse=True, sorted=True)
        total_scores = [total_scores[query_ridxs_idx == i]
                        for i in range(len(query_ridxs))]
        total_labels = [total_labels[query_ridxs_idx == i]
                        for i in range(len(query_ridxs))]
        ndcg30s = [ndcg(total_scores[i], total_labels[i], k=30).item()
                   for i in range(len(query_ridxs))]
        return val_loss / num_batches, np.mean(ndcg30s)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        total_scores, total_query_ridxs, total_candidate_ridxs = [], [], []
        for batch in tqdm(self.loaders['test']):
            batch = batch.to(self.device)
            scores = self.model(batch.query_inputs,
                                batch.candidate_inputs)
            total_scores.append(scores.flatten().detach().cpu())
            total_query_ridxs.append(
                batch.query_ridxs.flatten().detach().cpu())
            total_candidate_ridxs.append(
                batch.candidate_ridxs.flatten().detach().cpu())
        total_scores = torch.cat(total_scores)
        total_query_ridxs = torch.cat(total_query_ridxs)
        total_candidate_ridxs = torch.cat(total_candidate_ridxs)
        query_ridxs, query_ridxs_idx = torch.unique(
            total_query_ridxs, return_inverse=True, sorted=True)
        total_scores = [total_scores[query_ridxs_idx == i]
                        for i in range(len(query_ridxs))]
        total_candidate_ridxs = [total_candidate_ridxs[query_ridxs_idx == i]
                                 for i in range(len(query_ridxs))]
        predictions, scores = {}, {}
        for idx, query_ridx in enumerate(query_ridxs):
            predictions[str(query_ridx.item())] = total_candidate_ridxs[idx][torch.argsort(
                total_scores[idx], descending=True)][:30].tolist()
            predictions[str(query_ridx.item())] = [str(x)
                                                   for x in predictions[str(query_ridx.item())]]
            scores[str(query_ridx.item())] = {str(k.item()): v.item() for k, v in zip(
                total_candidate_ridxs[idx], total_scores[idx])}
        self.logger.write_predictions(predictions)
        self.logger.write_scores(scores)

    def train(self):
        for self.epoch in range(0, self.config.train.num_epochs + 1):
            train_loss = self.train_epoch() if self.epoch > 0 else 0
            val_loss, val_metric = self.valid()
            epoch_dict = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
            }
            self.logger.write_epoch(epoch_dict)
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch_dict = epoch_dict
                self.logger.dump_model(self.model)
                self.test()
        self.logger.write_final({
            "best_val_metric": self.best_val_metric,
            **self.best_epoch_dict
        })
