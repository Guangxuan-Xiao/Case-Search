from collections import defaultdict
import json
from icecream import ic
from evaluator import ndcg_with_rank
import numpy as np

def stat_list(list_):
    return {
        'min': min(list_),
        'max': max(list_),
        'mean': np.mean(list_),
        'std': np.std(list_),
        'len': len(list_)
    }

def check_ncdg(rank_dict, label_dict):
    ncdgs = []
    for query, ranks in rank_dict.items():
        # ic(query)
        labels = label_dict[query]
        # ic(sorted(labels.items(), key=lambda x: x[1], reverse=True))
        ncdg = ndcg_with_rank(ranks, labels, 30)
        # ic(ncdg)
        ncdgs.append(ncdg)
    ic(stat_list(ncdgs))


stage1_score_file = '/home/xgx/search-engine-project/log/retriever1-512-512-bi/1/val-scores.json'
stage2_score_file = '/home/xgx/search-engine-project/log/retriever2-512-512-bi/2/val-scores.json'
stage3_score_file = '/home/xgx/search-engine-project/log/retriever3-512-512-bi/1/val-scores.json'
label_file = '/home/xgx/search-engine-project/data/origin/val/label_top30_dict.json'
with open(stage1_score_file, 'r') as f:
    stage1_scores = json.load(f)
with open(stage2_score_file, 'r') as f:
    stage2_scores = json.load(f)
with open(stage3_score_file, 'r') as f:
    stage3_scores = json.load(f)
with open(label_file, 'r') as f:
    label_dict = json.load(f)
scores_dicts = [stage1_scores, stage2_scores, stage3_scores]
queries = list(stage1_scores.keys())[:]
# ic(queries)
current_ranks = {query: list(stage1_scores[query].keys()) for query in queries}
current_lens = {k: len(v) for k, v in current_ranks.items()}
for stage, current_scores in enumerate(scores_dicts[:]):
    ic(stage)
    for query, candidates in current_ranks.items():
        pairs = [(c, current_scores[query][c])
                 for c in candidates[:current_lens[query]]]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        current_lens[query] = 0
        for i, (c, s) in enumerate(sorted_pairs):
            current_ranks[query][i] = c
            if s >= 0.5:
                current_lens[query] += 1
    check_ncdg(current_ranks, label_dict)
