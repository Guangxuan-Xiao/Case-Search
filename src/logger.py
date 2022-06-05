import wandb
import os.path as osp
import os
import yaml
import json
import torch


class Logger:
    def __init__(self, config, seed):
        os.makedirs(osp.join(config.log_dir, config.title), exist_ok=True)
        with open(osp.join(config.log_dir, config.title, 'config.yml'), 'w+') as f:
            f.write(config.toYAML())
        self.log_dir = osp.join(config.log_dir, config.title, str(seed))
        os.makedirs(self.log_dir, exist_ok=True)
        self.use_wandb = config.use_wandb

    def write_step(self, stat):
        if self.use_wandb:
            wandb.log(stat)

    def write_epoch(self, stat):
        with open(osp.join(self.log_dir, 'log.json'), 'a') as f:
            f.write('{}\n'.format(stat))
        print(stat)
        if self.use_wandb:
            wandb.log(stat)

    def write_final(self, stat):
        with open(osp.join(self.log_dir, 'final.json'), 'a') as f:
            print(file=f)
            json.dump(stat, f)
        print(stat)
        if self.use_wandb:
            wandb.log(stat)

    def dump_model(self, model):
        torch.save(model.state_dict(), osp.join(self.log_dir, 'model.pth'))

    def write_predictions(self, predictions):
        with open(osp.join(self.log_dir, 'predictions.json'), 'w+') as f:
            json.dump(predictions, f)

    def write_scores(self, scores):
        with open(osp.join(self.log_dir, 'scores.json'), 'w+') as f:
            json.dump(scores, f)
