import torch
import torch.nn as nn
from spacecutter.models import LogisticCumulativeLink
from spacecutter.losses import CumulativeLinkLoss


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalRegressionLoss, self).__init__()
        self.link = LogisticCumulativeLink(num_classes)
        self.loss = CumulativeLinkLoss()

    def forward(self, score, label):
        # score: (batch_size, 1), label: (batch_size, 1)
        pred = self.link(score)
        # pred: (batch_size, num_classes)
        loss = self.loss(pred, label)
        return loss


class MSELoss(nn.Module):
    def __init__(self, num_classes):
        super(MSELoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.MSELoss()

    def forward(self, score, label):
        # score: (batch_size, 1), label: (batch_size, 1)
        loss = self.loss(score, label.float() / (self.num_classes - 1))
        return loss


def create_loss_fn(config):
    if config.model.loss_fn == 'OrdinalRegressionLoss':
        return OrdinalRegressionLoss(config.data.num_classes)
    elif config.model.loss_fn == 'MSELoss':
        return MSELoss(config.data.num_classes)
    else:
        raise ValueError("Unknown loss type {}".format(config.model.loss_fn))
