import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from numpy import genfromtxt
from transformers import AutoModel, AutoModelForSequenceClassification
from loss import create_loss_fn


class ConcatMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ConcatMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size * 2, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


def cls_pooling(x, mask):
    return x[:, 0, :]


def mean_pooling(x, mask):
    mask_expanded = mask.unsqueeze(-1).expand_as(x).float()
    sum_embeddings = (x * mask_expanded).sum(dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class TwoEncoder(torch.nn.Module):
    def __init__(self, config):
        super(TwoEncoder, self).__init__()
        self.encoder1 = AutoModel.from_pretrained(config.model.encoder)
        self.encoder2 = AutoModel.from_pretrained(config.model.encoder)
        if config.model.head == 'cosine':
            self.head = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif config.model.head == 'concat':
            self.head = ConcatMLP(self.encoder.config.hidden_size,
                                  32, 1, num_layers=2)
        else:
            raise ValueError(f'Unknown head: {config.model.head}')
        if config.model.pooler == 'mean':
            self.pooler = mean_pooling
        elif config.model.pooler == 'cls':
            self.pooler = cls_pooling
        else:
            raise ValueError(f'Unknown pooler: {config.model.pooler}')
        self.loss_fn = create_loss_fn(config)

    def forward(self, batch):
        query, candidate = batch.query_inputs, batch.candidate_inputs
        query_embeddings = self.encoder1(**query)['last_hidden_state']
        query_embedding = self.pooler(
            query_embeddings, query['attention_mask'])
        candidate_embeddings = self.encoder2(**candidate)['last_hidden_state']
        candidate_embedding = self.pooler(
            candidate_embeddings, candidate['attention_mask'])
        scores = self.head(query_embedding, candidate_embedding)
        if batch.labels.numel() > 0:
            loss = self.loss_fn(scores.reshape(-1, 1), batch.labels)
            return scores, loss
        return scores


class BiEncoder(torch.nn.Module):
    def __init__(self, config):
        super(BiEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.model.encoder)
        if config.model.head == 'cosine':
            self.head = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif config.model.head == 'concat':
            self.head = ConcatMLP(self.encoder.config.hidden_size,
                                  32, 1, num_layers=2)
        else:
            raise ValueError(f'Unknown head: {config.model.head}')
        if config.model.pooler == 'mean':
            self.pooler = mean_pooling
        elif config.model.pooler == 'cls':
            self.pooler = cls_pooling
        else:
            raise ValueError(f'Unknown pooler: {config.model.pooler}')
        self.loss_fn = create_loss_fn(config)

    def forward(self, batch):
        query, candidate = batch.query_inputs, batch.candidate_inputs
        query_embeddings = self.encoder(**query)['last_hidden_state']
        query_embedding = self.pooler(
            query_embeddings, query['attention_mask'])
        candidate_embeddings = self.encoder(**candidate)['last_hidden_state']
        candidate_embedding = self.pooler(
            candidate_embeddings, candidate['attention_mask'])
        scores = self.head(query_embedding, candidate_embedding)
        if batch.labels.numel() > 0:
            loss = self.loss_fn(scores.reshape(-1, 1), batch.labels)
            return scores, loss
        return scores


class CrossEncoder(torch.nn.Module):
    def __init__(self, config):
        super(CrossEncoder, self).__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            config.model.encoder, num_labels=1, ignore_mismatched_sizes=True)
        self.loss_fn = create_loss_fn(config)

    def forward(self, batch):
        input_dict = batch.concat_inputs
        x = self.encoder(**input_dict)
        logits = x['logits']
        scores = torch.sigmoid(logits)
        if batch.labels.numel() > 0:
            loss = self.loss_fn(logits, batch.labels)
            return scores, loss
        return scores


def create_model(config):
    if config.model.type == 'bi':
        return BiEncoder(config)
    elif config.model.type == 'cross':
        return CrossEncoder(config)
    elif config.model.type == 'two':
        return TwoEncoder(config)
    else:
        raise ValueError(f'Unknown model type: {config.model.type}')
