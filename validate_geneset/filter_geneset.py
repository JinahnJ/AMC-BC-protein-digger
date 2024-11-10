from typing import Tuple
import numpy as np
from generate_genepool.genepool_generator import tumor_pool, stroma_pool
from generate_genepool.mean_gene_pool import tumor_mean_pool, stroma_mean_pool
from generate_genepool.genepool_generator import generate_geneset
from validate_geneset.get_geneset import Genepool_container, Genepool_ranker, generate_genepool_cls, \
    p_generate_genepool_df_loader, Genepool_df_loader, Genepool_rank_result_container


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


def filter_acc_p_value_from_ranker(s: Genepool_ranker):
    perf = s.gene_pool_and_prediction_performance
    get_genpool_rank_container = lambda x: next(x[1])
    get_p_value = lambda x: x.ttest_p_value
    get_cm = lambda x: x.cm
    p = tuple(map(get_genpool_rank_container, perf))
    q = tuple(map(get_cm, p))
    r = tuple(map(accuracy_check, q))
    s = tuple(map(get_p_value, p))
    return p, q, r, s


if __name__ == '__main__':
    a = tuple([tumor_pool(), stroma_pool(), generate_geneset(stroma_mean_pool()), generate_geneset(tumor_mean_pool())])
    generate_genepool_tup = generate_genepool_cls(Genepool_container, *a)
    df_loader = tuple([p_generate_genepool_df_loader(i) for i in generate_genepool_tup])
    train_data = df_loader[0]
    s = Genepool_ranker(train_data, 2, 5, './config/config.yaml')
    q = filter_acc_p_value_from_ranker(s)
