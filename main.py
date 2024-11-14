import numpy as np
from utils.input import config_file
from utils.output import formatting_performance_and_model, save_result
from utils.dataframe_utils import get_dataframe
from generate_genepool.genepool_generator import generate_geneset, get_genepool
from generate_genepool.mean_gene_pool import get_mean_genepool
from validate_geneset.get_geneset import prediction_validation, Genepool_container, Genepool_rank_result_container, \
    generate_genepool_cls, Genepool_ranker, Genepool_df_loader
from validate_geneset.filter_geneset import filt_pred_val, get_acc_p_value_from_ranker
from model.model import LR_model, DT_model
import argparse
from itertools import product, chain
from functools import partial, reduce


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c",
                        type=str,
                        help="config file path, usually ./config/config.yaml")
    parser.add_argument("--result_path", "-r",
                        type=str,
                        help="result file path, usually ./src/result.yaml")
    pars_args = parser.parse_args()
    pars_args = vars(pars_args)

    config_dict = config_file(pars_args['config_path'])

    tumor_pool = get_genepool(pars_args['config_path'], config_dict['threshold']['Tumor'], 'Tumor')
    stroma_pool = get_genepool(pars_args['config_path'], config_dict['threshold']['Stroma'], 'Stroma')
    tumor_mean_pool = generate_geneset(get_mean_genepool(pars_args['config_path'], config_dict['threshold']['Tumor_mean'], ('Tumor',)))
    stroma_mean_pool = generate_geneset(get_mean_genepool(pars_args['config_path'], config_dict['threshold']['Stroma_mean'], ('Stroma',)))

    gene_pool = generate_genepool_cls(Genepool_container, tumor_pool, stroma_pool, tumor_mean_pool, stroma_mean_pool)

    lr = LR_model(config_dict_root=pars_args['config_path'], random_state=np.random.randint(10000))
    dt = DT_model(config_dict_root=pars_args['config_path'], random_state=np.random.randint(10000))
    df = get_dataframe(config_dict['dataset_file'])

    model_tup = lr, dt
    model_tup_product = tuple(product(model_tup, repeat=2))
    sfs_feature_selector_estimator = lambda x: x[0]
    validation_training_estimator = lambda x: x[1]

    pred_val_fc_tup = tuple([partial(prediction_validation, total_df=df, feature_selector_model=sfs_feature_selector_estimator(x),
                                     predictor_model=validation_training_estimator(x),
                                     config_dict_root=pars_args['config_path']) for x in model_tup_product])


    get_acc_p_val = partial(get_acc_p_value_from_ranker, config_dict=config_dict)

    acc_p_val_tup = tuple([tuple(map(pred_val_fc_tup[i], gene_pool)) for i in range(len(pred_val_fc_tup))])

    filt = lambda x : filt_pred_val(x, config_dict)
    filt_tup = tuple([tuple(map(filt,x)) for x in acc_p_val_tup])

    get_non_scaled = lambda x: x[0]
    get_std_scaled = lambda x: x[1]

    non_scaled_tup = tuple(tuple(map(get_non_scaled, x)) for x in filt_tup)
    std_scaled_tup= tuple(tuple(map(get_std_scaled, x)) for x in filt_tup)

    remove_blank_tuple = lambda x: reduce(chain, tuple(reduce(chain,x,())), ())
    non_scaled_tup_trimmed = tuple(remove_blank_tuple(non_scaled_tup))
    std_scaled_tup_trimmed = tuple(remove_blank_tuple(std_scaled_tup))

    non_scaled_result = formatting_performance_and_model(non_scaled_tup_trimmed, scaled=None)
    std_scaled_result = formatting_performance_and_model(std_scaled_tup_trimmed, scaled='std')
    result = list(enumerate((non_scaled_result or []) + (std_scaled_result or [])))
    save_result(result, pars_args['result_path'])


