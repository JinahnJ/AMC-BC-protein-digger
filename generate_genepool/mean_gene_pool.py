'''
Get mean ranking of each antibody
'''

from ranker.get_antibody_ranking import get_antibody_ranking
from functools import partial, reduce
from itertools import chain
import collections

def get_mean_ranking(t:tuple, key:tuple)->tuple:
    match = lambda x,y=0 : tuple([i for i in x if i[1] == y])
    match_p = partial(match, y=key)
    filter_t = map(match_p, t)
    rankings = map(lambda x: tuple([i[3:5] for i in x]) ,filter_t)
    rankings_flatten = reduce(chain, reduce(chain, rankings))

    scores = collections.defaultdict(list)
    for i in rankings_flatten:
        for j, k in i:
            scores[k].append(j)
    result_scores = {}
    for k, v in scores.items():
        result_scores[k] = sum(v) / len(v)
    result_scores = sorted(result_scores.items(), key=lambda x: x[1])

    result_dict = dict(result_scores)
    result_zip = zip(result_dict.values(), result_dict.keys())
    result_zip2 = zip(result_dict.values(), result_dict.keys())
    result_tup = tuple([
        ('mean',),
        key,
        ('LR_DT',),
        result_zip,
        result_zip2,
    ])

    return result_tup

def slicing_by_idx(t:tuple, idx:int)->tuple:
    f = lambda x, y : tuple(x)[:y]
    result_tuple = tuple([(t[0], t[1], t[2], f(t[3], idx), f(t[4], idx))])
    return result_tuple

def get_mean_genepool(root:str, threshold:int, key:tuple):
    rank = get_antibody_ranking(root)
    key_rank = get_mean_ranking(rank, key)
    sliced_key_rank = slicing_by_idx(key_rank, threshold)
    return sliced_key_rank

tumor_mean_pool = partial(get_mean_genepool, './config/config.yaml', 9, ('Tumor',))
stroma_mean_pool = partial(get_mean_genepool, './config/config.yaml', 19, ('Stroma',))


if __name__ == '__main__':
    a = get_antibody_ranking('./config/config.yaml')
    b = get_mean_ranking(a, ('Tumor',))
    c = slicing_by_idx(b, 10)