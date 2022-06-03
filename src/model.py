import torch
import torch.nn.functional as F
from icecream import ic
from numpy import genfromtxt
from transformers import AutoModel


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.model.encoder)

    def forward(self, query, candidate):
        query_embeddings = self.encoder(**query)['pooler_output']
        candidate_embeddings = self.encoder(**candidate)['pooler_output']
        return torch.cosine_similarity(query_embeddings, candidate_embeddings)


def create_model(config):
    return Model(config)
