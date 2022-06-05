import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from numpy import genfromtxt
from transformers import AutoModel


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


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
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

    def forward(self, query, candidate):
        query_embeddings = self.encoder(**query)['last_hidden_state']
        query_embedding = self.pooler(query_embeddings, query['attention_mask'])
        candidate_embeddings = self.encoder(**candidate)['last_hidden_state']
        candidate_embedding = self.pooler(candidate_embeddings, candidate['attention_mask'])
        return self.head(query_embedding, candidate_embedding)


def create_model(config):
    return Model(config)
