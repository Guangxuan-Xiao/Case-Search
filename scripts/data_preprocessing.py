import os
import os.path as osp
import numpy as np
import json
import torch
import re
from tqdm import tqdm
from icecream import ic
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../models/chinese-roberta-wwm-ext")
crime_pattern = re.compile(r'已构成(.*?)罪')


def preprocess_data(data_path, has_label=True):
    query_path = osp.join(data_path, 'query.json')
    candidates_path = osp.join(data_path, 'candidates')
    if has_label:
        label_path = osp.join(data_path, 'label_top30_dict.json')
        label = json.load(open(label_path))
        labels = []
    edges, inputs, query_ridxs, node_graph_ids, edge_graph_ids, candidate_ridxs = [
    ], [], [], [], [], []
    with open(query_path) as f:
        query_lines = f.readlines()
    for query_line in tqdm(query_lines):
        query_line = query_line.strip()
        query_dict = json.loads(query_line)
        input_str = "[SEP]".join(query_dict['crime']) + \
            "[SEP]"+query_dict['q']
        tokenized_inputs = tokenizer(input_str, return_tensors="pt")
        query_idx = len(inputs)
        inputs.append(tokenized_inputs)
        node_graph_ids.append(len(query_ridxs))
        query_ridxs.append(query_dict['ridx'])
        query_ridx = str(query_dict['ridx'])
        candidates_path = osp.join(candidates_path, query_ridx)
        for candidate in os.listdir(candidates_path):
            candidate_ridx = candidate[:-5]
            candidate_path = osp.join(candidates_path, candidate)
            candidate_dict = json.load(open(candidate_path))
            all_text = ''.join(candidate_dict.values())
            crime_name = crime_pattern.search(all_text)
            if crime_name is None:
                crime_name = ''
            else:
                crime_name = crime_name.group(1) + '罪'
            candidate_text = '[SEP]'.join(
                [crime_name, candidate_dict['ajjbqk']])
            tokenized_candidate = tokenizer(
                candidate_text, return_tensors="pt")
            candidate_idx = len(inputs)
            inputs.append(tokenized_candidate)
            node_graph_ids.append(node_graph_ids[-1])
            edge_graph_ids.append(node_graph_ids[-1])
            edges.append([query_idx, candidate_idx])
            candidate_ridxs.append(int(candidate_ridx))
            if has_label:
                if candidate_ridx in label[query_ridx]:
                    labels.append(label[query_ridx][candidate_ridx])
                else:
                    labels.append(0)
    if has_label:
        return edges, inputs, query_ridxs, node_graph_ids, edge_graph_ids, candidate_ridxs, labels
    return inputs, edges, query_ridxs, node_graph_ids, edge_graph_ids, candidate_ridxs


def save(processed_path, edges, inputs, query_ridxs, node_graph_ids, edge_graph_ids, candidate_ridxs, labels=None):
    torch.save(inputs, osp.join(processed_path, 'inputs.pt'))
    torch.save(edges, osp.join(processed_path, 'edges.pt'))
    torch.save(query_ridxs, osp.join(processed_path, 'query_ridxs.pt'))
    torch.save(candidate_ridxs, osp.join(processed_path, 'candidate_ridxs.pt'))
    torch.save(node_graph_ids, osp.join(processed_path, 'node_graph_ids.pt'))
    torch.save(edge_graph_ids, osp.join(processed_path, 'edge_graph_ids.pt'))
    if labels is not None:
        torch.save(labels, osp.join(processed_path, 'labels.pt'))


train_path = '../data/train'
train_inputs, train_edges, train_query_ridxs, train_node_graph_ids, train_edge_graph_ids, train_candidate_ridxs, train_labels = preprocess_data(
    train_path, has_label=True)

train_processed_path = '../data/train/processed'
save(train_processed_path, train_edges, train_inputs, train_query_ridxs,
     train_node_graph_ids, train_edge_graph_ids, train_candidate_ridxs, train_labels)

test_path = '../data/test'
test_inputs, test_edges, test_query_ridxs, test_node_graph_ids, test_edge_graph_ids, test_candidate_ridxs = preprocess_data(
    test_path, has_label=False)

test_processed_path = '../data/test/processed'