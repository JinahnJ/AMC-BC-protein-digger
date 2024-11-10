from typing import Any
import pandas as pd
import itertools
from .paired_df_dcls import Paired_dataframe
'''
generate iterator composed of permutation with factors intended.
'''
def generate_pair_tuple(grouping_tuple:tuple)-> tuple[Any] | Any:
    grouping_single = grouping_tuple
    if type(grouping_single) is str:
        r = grouping_single,
        return r
    else:
        pair_tuple = (itertools.permutations(grouping_single, n) for n in range(1,len(grouping_single)+1))
        flatten_tuple = itertools.chain.from_iterable(pair_tuple)
        return flatten_tuple

def get_dataframe(root='./dst/tidy_dataset.pkl' ):
     df = pd.read_pickle(root)
     return df

#generate tuple of Paired_dataframe, such as (Dataclass, Dataclass, Dataclass...)
def generate_pair_dataframe(iterable_divider, dataframe, d_cls,):
    df = dataframe
    names = list(iterable_divider)
    df_groupby_tup = tuple([df.groupby(list(name), observed=False) for name in names])
    d_cls_tup = (d_cls(name=n, df_group_tuple=df_g) for n, df_g in zip(names, df_groupby_tup))


    return d_cls_tup

if __name__ == '__main__':
    pass














