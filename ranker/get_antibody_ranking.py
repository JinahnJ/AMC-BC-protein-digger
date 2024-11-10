from typing import Tuple, Any, Generator, List

from utils.input import config_file
from utils.dataframe_utils import generate_pair_tuple, get_dataframe, generate_pair_dataframe
from utils.paired_df_dcls import Paired_dataframe, RFECV_result
from model.model import LR_model, DT_model

def get_antibody_ranking(config_dict_root: str) -> tuple[
    tuple[tuple[tuple, tuple, object, Any, Any], ...], tuple[tuple[tuple, tuple, object, Any, Any], ...]]:
    config_dict = config_file(config_dict_root)
    grouping_tuple = config_dict.get('target_category', ('cell_type',))
    target_tuple_pair = generate_pair_tuple(grouping_tuple)

    df_root = config_dict.get('dataset_file', '../dst/tidy_dataset.pkl')
    df = get_dataframe(df_root)

    paired_dataframe = generate_pair_dataframe(target_tuple_pair, df, Paired_dataframe)

    lr_model = LR_model(config_dict_root)
    dt_model = DT_model(config_dict_root)

    pivoted_dfs = (i.pivoted_df for i in paired_dataframe)
    unpack_pivoted_dfs = []
    for i in pivoted_dfs:
        for j, k, l, m in i:
            unpack_pivoted_dfs.append((j, k, l, m))
    unpack_pivoted_dfs = tuple(unpack_pivoted_dfs)

    lr_RFECV = tuple((
        RFECV_result(divider_name, group_name, x, y, lr_model)\
        for divider_name, group_name, x, y in unpack_pivoted_dfs
    ))
    dt_RFECV = tuple((
        RFECV_result(divider_name, group_name, x, y, dt_model)\
        for divider_name, group_name, x, y in unpack_pivoted_dfs
    ))
    # slice_rank_list = lambda x: RFECV_result.slicing_rank_list(x,5)
    slice_rank_list = lambda x : x
    lr_result_tup = tuple([
        (lr.divider_name, lr.group_name, lr.model, slice_rank_list(lr.rank_list),
         slice_rank_list(lr.std_rank_list)) for lr in lr_RFECV
    ])

    dt_result_tup = tuple([
        (dt.divider_name, dt.group_name, dt.model, slice_rank_list(dt.rank_list),
         slice_rank_list(dt.std_rank_list)) for dt in dt_RFECV
    ])

    # return tuple(ranking_list)
    return lr_result_tup, dt_result_tup

if __name__ == '__main__':
    a = get_antibody_ranking('./config/config.yaml')