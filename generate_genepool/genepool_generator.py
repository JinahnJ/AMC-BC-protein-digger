'''
Generate genepool (or gene sets) from ranked antibody (get_antibody_ranking)
''',
from ranker.get_antibody_ranking import get_antibody_ranking
from itertools import  chain, starmap
from functools import partial

#recieve tuple of gene ranking data
def filter_ranking(ranking:tuple, *args)->tuple:
    gen = chain.from_iterable(tuple(ranking))
    gen_ = (i for i in gen if i[1] == tuple(args))
    return tuple(gen_)

def slice_ranking(ranking:tuple, threshold:int)->tuple:
    f = lambda x, y : tuple([i for i in x if i[0]<y+1])
    slice_tuple = tuple([(i[0], i[1], i[2], f(i[3], threshold), f(i[4], threshold)) for i in ranking])
    return slice_tuple


def generate_geneset(t:tuple)->tuple:
    f = lambda x,y : y

    result_tuple = tuple([(i[0], i[1], i[2], set(starmap(f, i[3])), set(starmap(f,i[4]))) for i in t])
    return result_tuple

def get_genepool(root:str, threshold:int, *filter_args)->tuple:
    ab_ranking = get_antibody_ranking(root)
    filtered_ranking = filter_ranking(ab_ranking, *filter_args)
    sliced_ranking = slice_ranking(filtered_ranking, threshold)
    geneset = generate_geneset(sliced_ranking)
    return geneset

# tumor_pool = partial(get_genepool, './config/config.yaml', 1, 'Tumor')
# stroma_pool = partial(get_genepool, './config/config.yaml', 10, 'Stroma')

if __name__ == '__main__':
    a = get_antibody_ranking('./config/config.yaml')
    b = filter_ranking(a, 'Tumor')
    c = filter_ranking(a, 'Stroma')
    d = filter_ranking(a,  'Validation')

    e = slice_ranking(b, 1)
    f = generate_geneset(e)
    g = slice_ranking(c, 5)
    tumor_tree_vals = slice_ranking(c, 5)


