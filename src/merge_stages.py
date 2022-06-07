from collections import defaultdict
import json
from icecream import ic
from evaluator import ndcg_with_rank
import numpy as np
import os
import os.path as osp


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


def merge_stages(scores_dicts, label_dict=None):
    current_ranks = {query: list(d.keys())
                     for query, d in scores_dicts[0].items()}
    current_lens = {k: len(v) for k, v in current_ranks.items()}
    for stage, current_scores in enumerate(scores_dicts):
        for query, candidates in current_ranks.items():
            pairs = [(c, current_scores[query][c])
                     for c in candidates[:current_lens[query]]]
            sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
            current_lens[query] = 0
            for i, (c, s) in enumerate(sorted_pairs):
                current_ranks[query][i] = c
                if s >= 0.5:
                    current_lens[query] += 1
        if label_dict is not None:
            ic(stage + 1)
            check_ncdg(current_ranks, label_dict)
    return current_ranks


def valid():
    stage1_score_file = '../log/retriever1-512-512-bi-train/1/val-scores.json'
    stage2_score_file = '../log/retriever2-512-512-bi-train/1/val-scores.json'
    stage3_score_file = '../log/retriever3-512-512-bi-train/1/val-scores.json'
    label_file = '../data/origin/val/label_top30_dict.json'
    with open(stage1_score_file, 'r') as f:
        stage1_scores = json.load(f)
    with open(stage2_score_file, 'r') as f:
        stage2_scores = json.load(f)
    with open(stage3_score_file, 'r') as f:
        stage3_scores = json.load(f)
    with open(label_file, 'r') as f:
        label_dict = json.load(f)
    scores_dicts = [stage1_scores, stage2_scores, stage3_scores]

    current_ranks = merge_stages(scores_dicts, label_dict)


def test():
    stage1_score_file = '../log/retriever1-512-512-bi-train/1/test-scores.json'
    stage2_score_file = '../log/retriever2-512-512-bi-train/1/test-scores.json'
    stage3_score_file = '../log/retriever3-512-512-bi-train/1/test-scores.json'
    with open(stage1_score_file, 'r') as f:
        stage1_scores = json.load(f)
    with open(stage2_score_file, 'r') as f:
        stage2_scores = json.load(f)
    with open(stage3_score_file, 'r') as f:
        stage3_scores = json.load(f)
    scores_dicts = [stage1_scores, stage2_scores, stage3_scores]

    current_ranks = merge_stages(scores_dicts)
    for key in current_ranks.keys():
        current_ranks[key] = current_ranks[key][:30]
    json.dump(current_ranks, open(
        '../predictions/staged-single-model.json', 'w'))


def get_scores_list(path, prefix='test'):
    scores_list = []
    for seed in os.listdir(path):
        if not osp.isdir(osp.join(path, seed)) or seed in ['9', '14', '18', '22']:
            continue
        score_file = osp.join(path, seed, f'{prefix}-scores.json')
        if not osp.exists(score_file):
            continue
        with open(score_file, 'r') as f:
            scores = json.load(f)
        scores_list.append(scores)
    return scores_list


def ensemble_ranks(ranks_list):
    stat_ranks = defaultdict(dict)
    for ranks in ranks_list:
        for query, rank in ranks.items():
            for idx, ridx in enumerate(rank):
                if ridx in stat_ranks[query]:
                    stat_ranks[query][ridx].append(idx)
                else:
                    stat_ranks[query][ridx] = [idx]
    for query, d in stat_ranks.items():
        for ridx, ranks in d.items():
            stat_ranks[query][ridx] = stat_list(ranks)
        stat_ranks[query] = sorted(d.items(), key=lambda x: (
            x[1]['mean'], x[1]['std']))
    json.dump(stat_ranks, open(
        '../predictions/stat-ranks.json', 'w'))
    final_ranks = {}
    for query, d in stat_ranks.items():
        final_ranks[query] = list(map(lambda x: x[0], d))
    return final_ranks


def ensemble_test():
    stage1_path = '../log/retriever1-512-512-bi-train/'
    stage2_path = '../log/retriever2-512-512-bi-train/'
    stage3_path = '../log/retriever3-512-512-bi-train/'
    stage1_scores_list = get_scores_list(stage1_path)[:]
    stage2_scores_list = get_scores_list(stage2_path)
    stage3_scores_list = get_scores_list(stage3_path)
    ranks_list = []
    for stage1_scores in stage1_scores_list:
        for stage2_scores in stage2_scores_list:
            for stage3_scores in stage3_scores_list:
                scores_dicts = [stage1_scores, stage2_scores, stage3_scores]
                ranks = merge_stages(scores_dicts)
                ranks_list.append(ranks)
    ranks = ensemble_ranks(ranks_list)
    json.dump(ranks, open(
        '../predictions/staged-ensemble-model.json', 'w'))


def ensemble_scores(scores_list):
    agg_scores = defaultdict(dict)
    for scores_dict in scores_list:
        for query, scores in scores_dict.items():
            for candidate, score in scores.items():
                if candidate not in agg_scores[query]:
                    agg_scores[query][candidate] = score
                else:
                    agg_scores[query][candidate] += score
    for query, scores in agg_scores.items():
        for candidate in scores:
            agg_scores[query][candidate] /= len(scores_list)
    return agg_scores


def ensemble_test2():
    stage1_path = '../log/roberta-retriever1-train/'
    stage2_path = '../log/roberta-retriever2-train/'
    stage3_path = '../log/roberta-retriever3-train/'
    stage1_scores_list = get_scores_list(stage1_path)[:1]
    stage2_scores_list = get_scores_list(stage2_path)[:1]
    stage3_scores_list = get_scores_list(stage3_path)[:1]
    stage1_scores = ensemble_scores(stage1_scores_list)
    stage2_scores = ensemble_scores(stage2_scores_list)
    stage3_scores = ensemble_scores(stage3_scores_list)
    scores_dicts = [stage1_scores, stage2_scores, stage3_scores]
    ranks = merge_stages(scores_dicts)
    json.dump(ranks, open(
        '../predictions/roberta-3stage-ensemble-scores.json', 'w'))


if __name__ == '__main__':
    # ensemble_test()
    ensemble_test2()
    # valid()
    # test()
