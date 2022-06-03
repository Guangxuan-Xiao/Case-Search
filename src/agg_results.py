import json
import argparse
import os
import os.path as osp
from re import L
import numpy as np
from collections import defaultdict
from sklearn.metrics.cluster import adjusted_rand_score


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
        if 'group' in k:
            continue
        agg_res[k] = np.mean(v, axis=0).tolist()
        agg_res['{}_std'.format(k)] = np.std(v, axis=0).tolist()
    if 'in_group' in results and 'out_group' in results:
        in_groups, out_groups = results['in_group'], results['out_group']
        in_group_consensus, out_group_consensus = [], []
        for i in range(num_seeds):
            for j in range(i+1, num_seeds):
                in_group_consensus.append(adjusted_rand_score(
                    in_groups[i], in_groups[j]))
                out_group_consensus.append(adjusted_rand_score(
                    out_groups[i], out_groups[j]))
        agg_res['in_group_consensus'] = sum(
            in_group_consensus) / len(in_group_consensus)
        agg_res['in_group_consensus_std'] = np.std(in_group_consensus)
        agg_res['out_group_consensus'] = sum(
            out_group_consensus) / len(out_group_consensus)
        agg_res['out_group_consensus_std'] = np.std(out_group_consensus)
    with open(osp.join(log_dir, 'agg_results.json'), 'w+') as f:
        json.dump(agg_res, f, indent=4)
    return agg_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    args = parser.parse_args()
    agg_res = agg_results(args.log_dir)
