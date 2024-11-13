from functools import partial
from validate_geneset.get_geneset import prediction_validation, Genepool_container, Genepool_rank_result_container, \
    generate_genepool_cls, Genepool_ranker, Genepool_df_loader


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


def check_p_value(p_value_tuple: tuple, p_value_threshold:float=0.05) -> bool:
    p_value_btw_Rs, p_value_btw_NRs, p_value_in_train, p_value_in_val = p_value_tuple
    check_btw_Rs = more_than(p_value_btw_Rs, p_value_threshold)
    check_btw_NRs = more_than(p_value_btw_NRs, p_value_threshold)
    check_train = less_than(p_value_in_train, p_value_threshold)
    check_val = less_than(p_value_in_val, p_value_threshold)

    return all([check_btw_Rs, check_btw_NRs, check_train, check_val])


def accuracy_check(cm: tuple, train_accuracy_threshold:float=0.7, val_accuracy_threshold:float=0.7) -> bool:
    train_cm, val_cm = cm
    check_train_cm = more_than(train_cm[0], train_accuracy_threshold)
    check_val_cm = more_than(val_cm[0], val_accuracy_threshold)

    return all([check_train_cm, check_val_cm])


def check_accuracy_and_p_value(t: Genepool_rank_result_container) -> bool:
    p_value = check_p_value(t.ttest_p_value)
    cm = accuracy_check(t.cm)
    return all([p_value, cm])


def get_acc_p_value_from_ranker(t:tuple[Genepool_df_loader, Genepool_rank_result_container], config_dict:dict):
    get_gene_pool_df_loader = lambda x : x[0]
    get_gene_pool_rank_result_container = lambda x : x[1]
    cm = lambda x : get_gene_pool_rank_result_container(x).cm
    t_p_value = lambda x : get_gene_pool_rank_result_container(x).ttest_p_value
    config = config_dict['optimization_criteria']
    check_accuracy_ = partial(accuracy_check, train_accuracy_threshold=config.get('train_accuracy', 0.7), val_accuracy_threshold=config.get('val_accuracy', 0.7))
    check_p_value_ = partial(check_p_value, p_value_threshold=config.get('p-value', 0.05))
    return get_gene_pool_df_loader(t), get_gene_pool_rank_result_container(t), check_accuracy_(cm(t)), check_p_value_(t_p_value(t))

def filt_acc_p_value_from_ranker(t:tuple[Genepool_df_loader, Genepool_rank_result_container, bool, bool]):
    get_accuracy_pass_ = lambda x: x[2]
    get_p_value_pass_ = lambda x: x[3]
    return all([get_accuracy_pass_(t), get_p_value_pass_(t)])

def filt_pred_val(prediction_validatio_result, config_dict:dict):
    non_std_result, std_result = prediction_validatio_result
    get_accuracy_and_p_val = partial(get_acc_p_value_from_ranker, config_dict=config_dict)
    non_std_r_ = tuple(map(get_accuracy_and_p_val, non_std_result))
    std_r_ = tuple(map(get_accuracy_and_p_val, std_result))
    filt_non_std_r_ = tuple(filter(filt_acc_p_value_from_ranker, non_std_r_))
    filt_std_r_ = tuple(filter(filt_acc_p_value_from_ranker, std_r_))

    return filt_non_std_r_, filt_std_r_

if __name__ == '__main__':
    pass

