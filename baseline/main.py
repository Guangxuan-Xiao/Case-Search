import argparse
import json
import os
import random
from gensim.summarization import bm25
import jieba
import numpy as np

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str,
                    help='input path of the dataset directory.')
parser.add_argument('--output', type=str,
                    help='output path of the prediction file.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output

if __name__ == "__main__":
    print('begin...')
    result = {}
    with open(os.path.join(os.path.dirname(__file__), 'stopword.txt'), 'r') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.', '（', '）', '-'])

    lines = open(input_query_path, 'r').readlines()
    for line in lines:
        corpus = []
        query = str(eval(line)['ridx'])
        # model init
        result[query] = []
        files = os.listdir(os.path.join(input_candidate_path, query))
        for file_ in files:
            file_json = json.load(
                open(os.path.join(input_candidate_path, query, file_), 'r'))
            all_text = ''
            for key in ['ajjbqk' 'cpfxgc', 'pjjg', 'qw']:
                if key in file_json:
                    all_text += file_json[key]
            a = jieba.cut(all_text, cut_all=False)
            tem = " ".join(a).split()
            corpus.append([i for i in tem if not i in stopwords])
        bm25Model = bm25.BM25(corpus)

        # rank
        a = jieba.cut(eval(line)['q'], cut_all=False)
        tem = " ".join(a).split()
        q = [i for i in tem if not i in stopwords]
        raw_rank_index = np.array(
            bm25Model.get_scores(q)).argsort().tolist()[::-1]
        result[query] = [files[i].split('.')[0] for i in raw_rank_index]

    json.dump(result, open(output_path,
              "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
    print('ouput done.')
