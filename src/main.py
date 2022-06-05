import wandb
import torch
from utils import load_config, seed_all, create_optimizer, create_scheduler
from loss import create_loss_fn
from model import create_model
from data import create_loaders
from trainer import Trainer
from logger import Logger
from agg_results import agg_results
import os.path as osp

if __name__ == "__main__":
    config = load_config()
    print(config)
    for seed in config.seeds:
        if config.use_wandb:
            run = wandb.init(project="Law Search Engine",
                             config=config, name=config.title, reinit=True)
        seed_all(seed)
        logger = Logger(config, seed)
        device = torch.device(config.model.device)
        model = create_model(config)
        if len(config.model.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=config.model.device_ids)
        model.to(device)
        loaders = create_loaders(config)
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(optimizer, config)
        loss_fn = create_loss_fn(config)
        trainer = Trainer(model, loaders, optimizer, scheduler,
                          device, logger, loss_fn, config)
        trainer.train()
        if config.use_wandb:
            run.finish()
    agg_res = agg_results(osp.join(config.log_dir, config.title))
