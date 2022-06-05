import torch
import torch.nn as nn
from spacecutter.models import LogisticCumulativeLink
from spacecutter.losses import CumulativeLinkLoss
from icecream import ic

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


class OrdinalClassificationLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.BCEWithLogitsLoss()
        self.classifier = MLP(1, 50, num_classes, 2)

    def forward(self, score, label):
        # score: (batch_size, num_classes), label: (batch_size, 1)
        pred = self.classifier(score)
        expanded_label = torch.zeros_like(pred)
        for c in range(self.num_classes - 1):
            expanded_label[:, c] = (label == (c + 1)).flatten().float()
        loss = self.loss(pred, expanded_label)
        return loss

class BCELoss(nn.Module):
    def __init__(self, num_classes):
        super(BCELoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.BCELoss()

    def forward(self, score, label):
        # score: (batch_size, num_classes), label: (batch_size, 1)
        loss = self.loss(score, label.float() / (self.num_classes - 1))
        return loss

def create_loss_fn(config):
    if config.model.loss_fn == 'OrdinalRegressionLoss':
        return OrdinalRegressionLoss(config.data.num_classes)
    elif config.model.loss_fn == 'MSELoss':
        return MSELoss(config.data.num_classes)
    elif config.model.loss_fn == 'OrdinalClassificationLoss':
        return OrdinalClassificationLoss(config.data.num_classes)
    elif config.model.loss_fn == 'BCELoss':
        return BCELoss(config.data.num_classes)
    else:
        raise ValueError("Unknown loss type {}".format(config.model.loss_fn))
