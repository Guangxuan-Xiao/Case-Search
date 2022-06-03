from typing import NamedTuple
import torch
import numpy as np
import os.path as osp
from icecream import ic
from tqdm import trange

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


def pad_sequence(sequences, max_len, pad_value=0):
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
    return sequences


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
        for j in range(len(cluster_nodes)):
            for k in range(j + 1, len(cluster_nodes)):
                new_edges.append([cluster_nodes[j], cluster_nodes[k]])
                new_labels.append(3)
        other_labels = labels[i][labels[i] != 3]
        other_edges = edges[i][labels[i] != 3]
        for e, l in zip(other_edges, other_labels):
            for node in cluster_nodes:
                new_edges.append([node, e[1]])
                new_labels.append(l)
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
        self.split = split
        self.seq_len = config.data.seq_len

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        query_inputs = self.inputs[edge[0]]
        candidate_inputs = self.inputs[edge[1]]
        for key in query_inputs.keys():
            query_inputs[key] = pad_sequence(query_inputs[key], self.seq_len)
            candidate_inputs[key] = pad_sequence(
                candidate_inputs[key], self.seq_len)
        return_dict = {
            "query_inputs": self.inputs[edge[0]],
            "candidate_inputs": self.inputs[edge[1]],
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size,
                                         shuffle=shuffle, num_workers=config.data.num_workers, collate_fn=collate_fn)
    return loader


def create_loaders(config):
    loaders = {}
    for split in [config.mode, 'val', 'test']:
        loaders[split] = create_loader(config, split)
    return loaders
