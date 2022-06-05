from typing import NamedTuple
import torch
import numpy as np
import os.path as osp
from icecream import ic
from tqdm import trange
from copy import copy

class Batch(NamedTuple):
    query_inputs: dict
    candidate_inputs: dict
    query_ridxs: torch.Tensor
    candidate_ridxs: torch.Tensor
    labels: torch.Tensor

    def to(self, device):
        return Batch(
            {k: v.to(device) for k, v in self.query_inputs.items()},
            {k: v.to(device) for k, v in self.candidate_inputs.items()},
            self.query_ridxs.to(device),
            self.candidate_ridxs.to(device),
            self.labels.to(device),
        )


def pad_sequence(sequences, max_len, pad_value=0, is_inputs=False):
    # sequences: Tensor [batch_size, max_len]
    # max_len: int
    # pad_value: int
    current_len = sequences.size(1)
    if current_len < max_len:
        pad_len = max_len - current_len
        pad = torch.full((sequences.size(0), pad_len),
                         pad_value, dtype=sequences.dtype)
        sequences = torch.cat((sequences, pad), dim=1)
    else:
        sequences = sequences[:, :max_len]
        if is_inputs:
            sequences[:, -1] = 102
    return sequences


def crop_inputs(inputs, seq_len):
    input_ids = inputs['input_ids']
    current_len = input_ids.size(1)
    if current_len <= seq_len:
        return inputs
    crop_len = current_len - seq_len
    sep_indices = torch.nonzero(input_ids == 102)
    start_idx, end_idx = 1, current_len - 1 - crop_len
    if sep_indices.size(0) > 1:  # no crime
        start_idx = sep_indices[-2][1].item() + 1
    if start_idx >= end_idx:
        return inputs
    ret_inputs = {}
    select_idx = torch.randint(start_idx, end_idx, (1,)).item()
    input_ids = [input_ids[:, :select_idx],
                 input_ids[:, select_idx + crop_len:]]
    ret_inputs['input_ids'] = torch.cat(input_ids, dim=1)
    assert ret_inputs['input_ids'].size(1) == seq_len
    ret_inputs['attention_mask'] = inputs['attention_mask'][:, :seq_len]
    ret_inputs['token_type_ids'] = inputs['token_type_ids'][:, :seq_len]
    return ret_inputs


def augment_edges(edges, edges_graph_ids, labels):
    edges = np.array(edges)
    edges_graph_ids = np.array(edges_graph_ids)
    labels = np.array(labels)
    num_graphs = len(np.unique(edges_graph_ids))
    edges = [edges[edges_graph_ids == i] for i in range(num_graphs)]
    labels = [labels[edges_graph_ids == i] for i in range(num_graphs)]
    for i in trange(num_graphs):
        new_edges, new_labels = [], []
        cluster_nodes = [e[1] for e in edges[i][labels[i] == 3]]
        for node in cluster_nodes:
            edges_copy = edges[i][edges[i][:, 1] != node].copy()
            labels_copy = labels[i][edges[i][:, 1] != node].copy()
            edges_copy[:, 0] = node
            new_edges += edges_copy.tolist()
            new_labels += labels_copy.tolist()
        new_edges, new_labels = np.array(new_edges), np.array(new_labels)
        if new_edges.size > 0:
            edges[i] = np.concatenate([edges[i], new_edges])
            labels[i] = np.concatenate([labels[i], new_labels])
        edges[i], labels[i] = edges[i].tolist(), labels[i].tolist()
    edges = sum(edges, [])
    labels = sum(labels, [])
    return edges, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        path = osp.join(config.data.path, split, 'processed')
        self.edge_graph_ids = torch.load(
            osp.join(path, 'edge_graph_ids.pt'))  # E
        self.node_graph_ids = torch.load(
            osp.join(path, 'node_graph_ids.pt'))  # N
        self.edges = torch.load(osp.join(path, 'edges.pt'))  # E
        self.inputs = torch.load(osp.join(path, 'inputs.pt'))  # N
        self.query_ridxs = torch.load(osp.join(path, 'query_ridxs.pt'))  # G
        self.candidate_ridxs = torch.load(
            osp.join(path, 'candidate_ridxs.pt'))  # E
        if split != 'test':
            self.labels = torch.load(osp.join(path, 'labels.pt'))
        else:
            self.labels = None
        if 'train' in split and config.data.augment_edge:
            ic(f'Augmenting {split} edges, Before E = {len(self.edges)}')
            self.edges, self.labels = augment_edges(
                self.edges, self.edge_graph_ids, self.labels)
            ic(f'Augmented {split} edges, After E = {len(self.edges)}')
        self.crop_candidate = 'train' in split and config.data.crop_candidate
        self.split = split
        self.seq_len = config.data.seq_len

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        query_inputs = copy(self.inputs[edge[0]])
        candidate_inputs = copy(self.inputs[edge[1]])
        if self.crop_candidate:
            candidate_inputs = crop_inputs(candidate_inputs, self.seq_len)
        for key in query_inputs.keys():
            query_inputs[key] = pad_sequence(
                query_inputs[key], self.seq_len, is_inputs=key == 'inputs')
            candidate_inputs[key] = pad_sequence(
                candidate_inputs[key], self.seq_len, is_inputs=key == 'inputs')
        return_dict = {
            "query_inputs": query_inputs,
            "candidate_inputs": candidate_inputs,
            "query_ridx": 0 if 'train' in self.split else self.query_ridxs[self.edge_graph_ids[idx]],
            "candidate_ridx": 0 if 'train' in self.split else self.candidate_ridxs[idx],
            "label": self.labels[idx] if self.labels is not None else None
        }
        return return_dict


def collate_fn(batch):
    query_inputs = [item['query_inputs'] for item in batch]
    query_input_ids = torch.cat([item['input_ids']
                                for item in query_inputs], dim=0)
    query_attention_mask = torch.cat(
        [item['attention_mask'] for item in query_inputs], dim=0)
    query_inputs = {
        'input_ids': query_input_ids,
        'attention_mask': query_attention_mask
    }
    candidate_inputs = [item['candidate_inputs'] for item in batch]
    candidate_input_ids = torch.cat(
        [item['input_ids'] for item in candidate_inputs], dim=0)
    candidate_attention_mask = torch.cat(
        [item['attention_mask'] for item in candidate_inputs], dim=0)
    candidate_inputs = {
        'input_ids': candidate_input_ids,
        'attention_mask': candidate_attention_mask
    }
    query_ridxs = [item['query_ridx'] for item in batch]
    candidate_ridxs = [item['candidate_ridx'] for item in batch]
    labels = [item['label'] for item in batch]
    if None in labels:
        labels = []
    return Batch(query_inputs,
                 candidate_inputs,
                 torch.tensor(query_ridxs),
                 torch.tensor(candidate_ridxs),
                 torch.tensor(labels).reshape(-1, 1)
                 )


def create_loader(config, split):
    dataset = Dataset(config, split)
    shuffle = 'train' in split
    batch_size = config.train.batch_size if 'train' in split else config.train.eval_batch_size
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=config.data.num_workers, collate_fn=collate_fn)
    return loader


def create_loaders(config):
    loaders = {}
    for split in [config.mode, 'val', 'test']:
        loaders[split] = create_loader(config, split)
    return loaders
