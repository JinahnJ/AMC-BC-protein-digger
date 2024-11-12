'''
Take gene sets from genepool_generator and slice dataframe
'''
from itertools import chain, batched
from functools import partial
import pandas as pd
from dataclasses import dataclass, field
import sklearn.feature_selection
from utils.paired_df_dcls import Paired_dataframe
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_ind
import numpy as np
from utils.input import config_file
from model.model import SFS_setting

@dataclass(frozen=True)
class Genepool_container:
    divider_name : tuple
    group_name : tuple
    model : tuple
    non_scaled_pool : set
    std_pool : set

@dataclass(frozen=True)
class Train_val_xy:
    train_x: pd.DataFrame
    train_y: pd.Series
    val_x: pd.DataFrame
    val_y: pd.Series
    std_train_x: pd.DataFrame
    std_train_y: pd.Series
    std_val_x: pd.DataFrame
    std_val_y: pd.Series

def generate_genepool_cls(d_cls, *args):
    f = lambda x : d_cls(*x)
    result_t = tuple(tuple((map(f, i)) for i in args))
    result_t = tuple(chain.from_iterable(result_t))
    return result_t

@dataclass(frozen=True)
class Genepool_df_loader:
    container: Genepool_container
    df:pd.DataFrame
    sliced_df:pd.DataFrame
    std_df:pd.DataFrame
    train_val_df:tuple[tuple,tuple]
    load_train_val_data:Train_val_xy


def slice_df(df: pd.DataFrame, pool:set, column_name:str='Antibody' ) -> pd.DataFrame:
    df_ = df.copy()
    s = pool
    df2 = df_.loc[df_[column_name].isin(s)]
    return df2

def std_df(df:pd.DataFrame, column_name:str='Value') -> pd.DataFrame:
    df_ = df.copy()
    df_[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    return df_


def train_val_df(non_scaled_df:pd.DataFrame, sliced_std_df:pd.DataFrame,
                 target_column:str='train_group', train_msk:str='Train', val_msk:str='Validation')->tuple:
    get_train = lambda x : x.copy().loc[x[target_column] == train_msk]
    get_validation = lambda x: x.copy().loc[x[target_column] == val_msk]

    train_tuple = tuple(map(get_train, (non_scaled_df, sliced_std_df)))
    val_tuple = tuple(map(get_validation, (non_scaled_df, sliced_std_df)))

    return train_tuple, val_tuple

def load_train_val_data(t:tuple)->dataclass:
    f = Paired_dataframe.pivot_df
    flatten_t = chain.from_iterable(t)
    r = map(f, flatten_t)
    r_batch = batched(r, 2)
    r_batched_zip = tuple(zip(*r_batch))
    train_val_xy = Train_val_xy(r_batched_zip[0][0][0], r_batched_zip[0][0][1],
                                r_batched_zip[0][1][0], r_batched_zip[0][1][1],
                                r_batched_zip[1][0][0], r_batched_zip[1][0][1],
                                r_batched_zip[1][1][0], r_batched_zip[1][1][1],)
    return train_val_xy



def generate_genepool_df_loader(d_cls:Genepool_container, df:pd.DataFrame) -> Genepool_df_loader:
    df_ = df.copy()
    non_scaled_pool = d_cls.non_scaled_pool
    std_pool = d_cls.std_pool
    sliced_df = slice_df(df_, non_scaled_pool)
    scaled_df = std_df(df_)
    std_sliced_df = slice_df(scaled_df, std_pool)
    t_v_df = train_val_df(sliced_df, std_sliced_df)
    t_v_xy = load_train_val_data(t_v_df)

    genepool_loader_ins = Genepool_df_loader(d_cls, df_, sliced_df, std_sliced_df, t_v_df, t_v_xy)

    return genepool_loader_ins


@dataclass(frozen=True)
class Genepool_rank_result_container:
    base_model : object #scikitlearn classifier
    ttest_p_value : tuple #  p_value_btw_Rs, p_value_btw_NRs, p_value_in_train, p_value_in_val
    cm : tuple #(train accuracy, fdr, fpr), (val accuracy, fdr, fpr)
    antibody: str = field(init=False)  # This will be calculated in __post_init__
    base_model_: str = field(init=False)  # Calculated at initialization

    def __post_init__(self):
        # Calculate antibody and base_model_ properties at initialization
        object.__setattr__(self, 'antibody', self.base_model.feature_names_in_)
        object.__setattr__(self, 'base_model_', str(self.base_model))

@dataclass(frozen=True)
class Genepool_ranker:
    genepool_df_loader:Genepool_df_loader
    min_vals:int
    max_vals:int
    config_dict_root:str

def generate_genepool_ranker(genepool_df_loader:Genepool_df_loader, min_vals:int, max_vals:int, config_dict_root:str='./config/config.yaml') -> Genepool_ranker:
    return Genepool_ranker(genepool_df_loader, min_vals, max_vals, config_dict_root)

def sfs_tuple(non_scaled_pool:set, std_pool:set,
              min_vals:int, max_vals:int, estimator:object, config_dict_root:str) -> tuple:
    max_val = 0
    max_std_val = 0

    #check antibody length
    if max_vals >= len(non_scaled_pool):
        max_val = len(non_scaled_pool) - 1
    else: max_val = max_vals

    if max_vals >= len(std_pool):
        max_std_val = len(std_pool) - 1
    else: max_std_val = max_vals

    non_scaled_sfs_tup = tuple(
        (
            ((SequentialFeatureSelector(estimator, n_features_to_select=i,
                                       direction='forward',
                                       **SFS_setting(config_dict_root)),
             (SequentialFeatureSelector(estimator, n_features_to_select=i,
                                        direction='backward', **SFS_setting(config_dict_root))))
             for i in range(min_vals, max_val+1))
        )
    )
    std_sfs_tup = tuple(
        ((SequentialFeatureSelector(estimator, n_features_to_select=i,
                                   direction='forward', **SFS_setting(config_dict_root)),
         (SequentialFeatureSelector(estimator, n_features_to_select=i,
                                    direction='backward', **SFS_setting(config_dict_root))))
         for i in range(min_vals, max_std_val+1)

        )
    )
    return non_scaled_sfs_tup, std_sfs_tup

def generate_sfs_tuple(genepool_ranker:Genepool_ranker, estimator:object)->tuple:
    c = genepool_ranker.genepool_df_loader.container
    non_scaled_pool = c.non_scaled_pool
    std_pool = c.std_pool
    min_val = genepool_ranker.min_vals
    max_val = genepool_ranker.max_vals
    config_dict_root = genepool_ranker.config_dict_root
    return sfs_tuple(non_scaled_pool, std_pool, min_val, max_val, estimator, config_dict_root)


def trained_sfs(sfs:sklearn.feature_selection.SequentialFeatureSelector, train_x:pd.DataFrame,
                train_y:pd.Series,)-> object:
    trained_sfs = sfs.fit(train_x, train_y)
    return trained_sfs

def train_sfs(t:tuple[sklearn.feature_selection.SequentialFeatureSelector], ranker:Genepool_ranker)->tuple[sklearn.feature_selection.SequentialFeatureSelector,]:
    train_x = ranker.genepool_df_loader.load_train_val_data.train_x
    train_y = ranker.genepool_df_loader.load_train_val_data.train_y
    double_chain = lambda x : tuple(chain.from_iterable(tuple(chain.from_iterable(x))))
    f = partial(trained_sfs, train_x=train_x, train_y=train_y)
    return tuple(map(f, double_chain(t)))


def get_ranked_antibody(trained_sfs_models:sklearn.feature_selection.SequentialFeatureSelector, )->tuple:
    ranking = tuple([trained_sfs_models.feature_names_in_[trained_sfs_models.get_support()]])
    return ranking


def slicing_df(gene_list,tr_x, tr_y, val_x, val_y)->tuple:
    tup = tuple([tuple([tr_x[gene], tr_y, val_x[gene], val_y]) for gene in gene_list])
    return tup

def get_slice_df_by_ab(gene_ranker:Genepool_ranker, ab_rank:tuple[np.array]):
    f = partial(slicing_df, tr_x=gene_ranker.genepool_df_loader.load_train_val_data.train_x,
                tr_y=gene_ranker.genepool_df_loader.load_train_val_data.train_y,
                val_x=gene_ranker.genepool_df_loader.load_train_val_data.val_x,
                val_y=gene_ranker.genepool_df_loader.load_train_val_data.val_y)
    return tuple(chain.from_iterable(tuple(map(f, ab_rank))))

def get_performance(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fdr = fp / (fp+tp)
    fpr = fp / (fp+tn)
    return accuracy, fdr, fpr

def get_t_test_p_value(t_x:pd.DataFrame, t_y:pd.Series, v_x:pd.DataFrame, v_y:pd.Series) -> tuple:
    ttest = lambda x, y : ttest_ind(x, y).pvalue
    if isinstance(t_x, pd.DataFrame):
        t_y = t_y.reindex(t_x.index)
    if isinstance(v_x, pd.DataFrame):
        v_y = v_y.reindex(v_x.index)

    t_x = t_x.to_numpy() if isinstance(t_x, pd.DataFrame) else t_x
    v_x = v_x.to_numpy() if isinstance(v_x, pd.DataFrame) else v_x
    t_x_r = t_x[t_y == 1].flatten()
    t_x_nr = t_x[t_y == 0].flatten()
    v_x_r = v_x[v_y == 1].flatten()
    v_x_nr = v_x[v_y == 0].flatten()

    p_value_btw_Rs = ttest(t_x_r, v_x_r)
    p_value_btw_NRs = ttest(t_x_nr, v_x_nr)
    p_value_in_train = ttest(t_x_r, t_x_nr)
    p_value_in_val = ttest(v_x_r, v_x_nr)


    return p_value_btw_Rs, p_value_btw_NRs, p_value_in_train, p_value_in_val


def prediction_performance(t:tuple, estimator:object)->object:
    t_x, t_y, v_x, v_y = t
    train_m = estimator.fit(t_x, t_y)
    pred_train_x = train_m.predict(t_x)
    pred_val_x = train_m.predict(v_x)
    train_cm = confusion_matrix(t_y, pred_train_x)
    val_cm = confusion_matrix(v_y, pred_val_x)
    ttest_result = get_t_test_p_value(pred_train_x, t_y, pred_val_x, v_y)

    return Genepool_rank_result_container(train_m, ttest_result, tuple([get_performance(train_cm), get_performance(val_cm)]))


def gene_pool_and_prediction_performance(rank_result:Genepool_rank_result_container, df_loader:Genepool_df_loader):
    gene_pool = df_loader
    pred_performance = rank_result

    return gene_pool, pred_performance

def prediction_validation(gene_pool_container:Genepool_container, total_df:pd.DataFrame, feature_selector_model:object,
                          predictor_model:object, config_dict_root:str='./config/config.yaml'):
    config_dict = config_file(config_dict_root)
    gene_pool_df_loader = generate_genepool_df_loader(gene_pool_container, total_df)
    generate_gene_pool_ranker = partial(generate_genepool_ranker, min_vals=config_dict['genepool_selection_range']['min_value'],
                                 max_vals=config_dict['genepool_selection_range']['max_value'], config_dict_root=config_dict_root)
    gene_pool_ranker = generate_gene_pool_ranker(gene_pool_df_loader)
    generate_sfs = partial(generate_sfs_tuple, estimator=feature_selector_model)
    untrained_sfs = generate_sfs(gene_pool_ranker)
    trained_sfs_models = train_sfs(untrained_sfs, gene_pool_ranker)
    ranked_antibody = tuple(map(get_ranked_antibody, trained_sfs_models))
    sliced_df = get_slice_df_by_ab(gene_pool_ranker, ranked_antibody)
    predict_performance = partial(prediction_performance, estimator=predictor_model)
    perform = tuple(map(predict_performance, sliced_df))
    contain_df_loader = partial(gene_pool_and_prediction_performance, df_loader=gene_pool_ranker.genepool_df_loader)
    result = tuple(map(contain_df_loader, perform))

    return result



if __name__ == '__main__':
    pass
    # a = tuple([tumor_pool(), stroma_pool(), generate_geneset(stroma_mean_pool()), generate_geneset(tumor_mean_pool())])
    # generate_genepool_tup = generate_genepool_cls(Genepool_container, *a)
    # b = generate_genepool_tup[1]
    # config_dict = config_file('./config/config.yaml')
    # lr = LR_model(config_dict_root='./config/config.yaml', random_state=np.random.randint(10000))
    # dt = DT_model(config_dict_root='./config/config.yaml', random_state=np.random.randint(10000))
    # df = get_dataframe(config_dict['dataset_file'])
    # c = prediction_validation(b, df, lr, lr, './config/config.yaml')
    #
    # p_f = partial(prediction_validation, total_df=df, feature_selector_model=dt, predictor_model=dt, config_dict_root='./config/config.yaml')
    # t = tuple(map(p_f, generate_genepool_tup))
