from typing import Tuple
import numpy as np
from functools import partial
from utils.input import config_file
from generate_genepool.genepool_generator import generate_geneset
from validate_geneset.get_geneset import prediction_validation, Genepool_container, Genepool_rank_result_container, \
    generate_genepool_cls, Genepool_ranker, Genepool_df_loader
from model.model import LR_model, DT_model, SFS_setting
from utils.dataframe_utils import get_dataframe
from generate_genepool.genepool_generator import tumor_pool, stroma_pool
from generate_genepool.mean_gene_pool import tumor_mean_pool, stroma_mean_pool

def more_than(i: float, threshold: float) -> bool:
    if i >= threshold:
        return True
    else:
        return False


def less_than(i: float, threshold: float) -> bool:
    if i < threshold:
        return True
    else:
        return False


def check_p_value(p_value_tuple: tuple, p_value_threshold: float = 0.05) -> bool:
    p_value_btw_Rs, p_value_btw_NRs, p_value_in_train, p_value_in_val = p_value_tuple
    check_btw_Rs = more_than(p_value_btw_Rs, p_value_threshold)
    check_btw_NRs = more_than(p_value_btw_NRs, p_value_threshold)
    check_train = less_than(p_value_in_train, p_value_threshold)
    check_val = less_than(p_value_in_val, p_value_threshold)

    return all([check_btw_Rs, check_btw_NRs, check_train, check_val])


def accuracy_check(cm: tuple, accuracy_threshold: float = 0.7) -> bool:
    train_cm, val_cm = cm
    check_train_cm = more_than(train_cm[0], accuracy_threshold)
    check_val_cm = more_than(val_cm[0], accuracy_threshold)

    return all([check_train_cm, check_val_cm])


def check_accuracy_and_p_value(t: Genepool_rank_result_container) -> bool:
    p_value = check_p_value(t.ttest_p_value)
    cm = accuracy_check(t.cm)
    return all([p_value, cm])


def filter_acc_p_value_from_ranker(t:tuple[Genepool_df_loader, Genepool_rank_result_container]):
    get_gene_pool_df_loader = lambda x : x[0]
    get_gene_pool_rank_result_container = lambda x : x[1]
    cm = lambda x : get_gene_pool_rank_result_container(x).cm
    t_p_value = lambda x : get_gene_pool_rank_result_container(x).ttest_p_value

    return get_gene_pool_df_loader(t), get_gene_pool_rank_result_container(t), accuracy_check(cm(t)), check_p_value(t_p_value(t))


if __name__ == '__main__':
    a = tuple([tumor_pool(), stroma_pool(), generate_geneset(stroma_mean_pool()), generate_geneset(tumor_mean_pool())])
    generate_genepool_tup = generate_genepool_cls(Genepool_container, *a)
    config_dict = config_file('./config/config.yaml')
    lr = LR_model(config_dict_root='./config/config.yaml', random_state=np.random.randint(10000))
    dt = DT_model(config_dict_root='./config/config.yaml', random_state=np.random.randint(10000))
    df = get_dataframe(config_dict['dataset_file'])
    p_f = partial(prediction_validation, total_df=df, feature_selector_model=dt, predictor_model=dt, config_dict_root='./config/config.yaml')
    t = tuple(map(p_f, generate_genepool_tup))
    b = t[1]
    c = tuple(map(filter_acc_p_value_from_ranker, b))
