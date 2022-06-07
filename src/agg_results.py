import json
import argparse
import os
import os.path as osp
from re import L
import numpy as np
from collections import defaultdict
from sklearn.metrics.cluster import adjusted_rand_score
from icecream import ic

def agg_results(log_dir):
    results = defaultdict(list)
    num_seeds = 0
    for seed in os.listdir(log_dir):
        seed_dir = osp.join(log_dir, seed)
        if not osp.isdir(seed_dir):
            continue
        final_file = osp.join(seed_dir, 'final.json')
        if not osp.isfile(final_file):
            continue
        with open(final_file) as f:
            for k, v in json.loads(f.readlines()[-1]).items():
                results[k].append(v)
        num_seeds += 1

    agg_res = {}
    for k, v in results.items():
        agg_res[k] = np.mean(v, axis=0).tolist()
        agg_res['{}_std'.format(k)] = np.std(v, axis=0).tolist()
    with open(osp.join(log_dir, 'agg_results.json'), 'w+') as f:
        json.dump(agg_res, f, indent=4)
    return agg_res


def agg_scores(log_dir):
    agg_scores = defaultdict(dict)
    num_seeds = 0
    for seed in os.listdir(log_dir):
        seed_dir = osp.join(log_dir, seed)
        if not osp.isdir(seed_dir):
            continue
        score_file = osp.join(seed_dir, 'test-scores.json')
        if not osp.isfile(score_file):
            continue
        with open(score_file) as f:
            for query, scores_dict in json.loads(f.readlines()[-1]).items():
                for candidate, score in scores_dict.items():
                    if candidate not in agg_scores[query]:
                        agg_scores[query][candidate] = score
                    else:
                        agg_scores[query][candidate] += score
        num_seeds += 1
    agg_rank = {}
    for query, scores_dict in agg_scores.items():
        agg_rank[query] = [c for c, _ in sorted(
            scores_dict.items(), key=lambda x: x[1], reverse=True)][:30]
    with open(osp.join(log_dir, 'agg_predictions.json'), 'w+') as f:
        json.dump(agg_rank, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    args = parser.parse_args()
    agg_res = agg_results(args.log_dir)
    agg_scores(args.log_dir)
