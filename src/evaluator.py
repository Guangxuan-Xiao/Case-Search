import torch
import os
import numpy as np
import json
import math
from tqdm import tqdm
from icecream import ic


def ndcg(scores, labels, k):
    dcg_value = 0.
    idcg_value = 0.

    sorted_labels = torch.sort(labels, descending=True)[0]
    sorted_preds = labels[torch.argsort(scores, descending=True)]

    for i in range(0, k):
        logi = math.log(i+2, 2)
        dcg_value += sorted_preds[i] / logi
        idcg_value += sorted_labels[i] / logi

    return dcg_value / idcg_value


def ndcg_with_rank(ranks, labels, k):
    # ranks: list of ridxs
    # labels: dict  {'ridx': score}
    dcg_value = 0.
    idcg_value = 0.
    sorted_labels = sorted(labels.values(), reverse=True)
    for i in range(0, k):
        logi = math.log(i+2, 2)
        label = labels.get(ranks[i], 0)
        dcg_value += label / logi
        idcg_value += sorted_labels[i] / logi

    return dcg_value / idcg_value
