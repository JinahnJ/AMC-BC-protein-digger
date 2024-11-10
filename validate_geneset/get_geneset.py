'''
Take gene sets from genepool_generator and slice dataframe
'''
from functools import partial
from itertools import chain, batched, repeat
import pandas as pd
from dataclasses import dataclass, field
from utils.dataframe_utils import get_dataframe
from utils.input import config_file
from utils.paired_df_dcls import Paired_dataframe
from sklearn.feature_selection import SequentialFeatureSelector
from model.model import LR_model, DT_model, SFS_setting
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_ind
import numpy as np

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

    @property
    def sliced_df(self):
        df_ = self.df.copy()
        s = self.container.non_scaled_pool
        df2 = df_.loc[df_['Antibody'].isin(s)]
        return df2

    @property
    def std_df(self):
        df_ = self.df.copy()
        df_['Value'] = (self.df['Value'] - self.df['Value'].mean()) / self.df['Value'].std()
        return df_

    @property
    def sliced_std_df(self):
        df_ = self.std_df.copy()
        s = self.container.std_pool
        df2 = df_.loc[df_['Antibody'].isin(s)]
        return df2

    @property
    def train_val_df(self)->tuple:
        get_train = lambda x : x.copy().loc[x['train_group'] == 'Train']
        get_validation = lambda x: x.copy().loc[x['train_group'] == 'Validation']
        train_tuple = tuple(map(get_train, (self.sliced_df, self.sliced_std_df)))
        val_tuple = tuple(map(get_validation, (self.sliced_df, self.sliced_std_df)))

        return train_tuple, val_tuple


    @property
    def load_train_val_data(self)->dataclass:
        f = Paired_dataframe.pivot_df
        flatten_t = chain.from_iterable(self.train_val_df)
        r = map(f, flatten_t)
        r_batch = batched(r, 2)
        r_batched_zip = tuple(zip(*r_batch))
        train_val_xy = Train_val_xy(r_batched_zip[0][0][0], r_batched_zip[0][0][1],
                                    r_batched_zip[0][1][0], r_batched_zip[0][1][1],
                                    r_batched_zip[1][0][0], r_batched_zip[1][0][1],
                                    r_batched_zip[1][1][0], r_batched_zip[1][1][1],)
        return train_val_xy



def generate_genepool_df_loader(d_cls:Genepool_container, df:pd.DataFrame) -> Genepool_df_loader:
    return Genepool_df_loader(d_cls, df)

p_generate_genepool_df_loader = partial(generate_genepool_df_loader
                                        ,df=get_dataframe(config_file('./config/config.yaml')['dataset_file']))

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
    config_dict_root:str #for LR, DT, sfs setting
    seed:int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'seed', np.random.randint(10000))

    @property
    def lr_estimator(self):
        lr = LR_model(self.config_dict_root, random_state=self.seed)
        return lr

    @property
    def dt_estimator(self):
        dt = DT_model(self.config_dict_root, random_state=self.seed)
        return dt

    @property
    def sfs_tuple(self):
        max_val = 0
        max_std_val = 0

        #check antibody length
        if self.max_vals >= len(self.genepool_df_loader.container.non_scaled_pool):
            max_val = len(self.genepool_df_loader.container.non_scaled_pool) - 1
        else: max_val = self.max_vals

        if self.max_vals >= len(self.genepool_df_loader.container.std_pool):
            max_std_val = len(self.genepool_df_loader.container.std_pool) - 1
        else: max_std_val = self.max_vals

        lr_sfs_tup = tuple(
            (
                ((SequentialFeatureSelector(self.lr_estimator, n_features_to_select=i,
                                           direction='forward',
                                           **SFS_setting(self.config_dict_root)),
                 (SequentialFeatureSelector(self.lr_estimator, n_features_to_select=i,
                                            direction='backward', **SFS_setting(self.config_dict_root))))
                 for i in range(self.min_vals, max_val+1))
            )
        )
        dt_sfs_tup = tuple(
            ((SequentialFeatureSelector(self.dt_estimator, n_features_to_select=i,
                                       direction='forward', **SFS_setting(self.config_dict_root)),
             (SequentialFeatureSelector(self.dt_estimator, n_features_to_select=i,
                                        direction='backward', **SFS_setting(self.config_dict_root))))
             for i in range(self.min_vals, max_val+1)

            )
        )
        lr_std_sfs_tup = tuple(
            (
                ((SequentialFeatureSelector(self.lr_estimator, n_features_to_select=i,
                                           direction='forward',
                                           **SFS_setting(self.config_dict_root)),
                 (SequentialFeatureSelector(self.lr_estimator, n_features_to_select=i,
                                            direction='backward', **SFS_setting(self.config_dict_root))))
                 for i in range(self.min_vals, max_std_val+1))
            )
        )
        dt_std_sfs_tup = tuple(
            ((SequentialFeatureSelector(self.dt_estimator, n_features_to_select=i,
                                       direction='forward', **SFS_setting(self.config_dict_root)),
             (SequentialFeatureSelector(self.dt_estimator, n_features_to_select=i,
                                        direction='backward', **SFS_setting(self.config_dict_root))))
             for i in range(self.min_vals, max_std_val+1)

            )
        )
        return lr_sfs_tup, dt_sfs_tup, lr_std_sfs_tup, dt_std_sfs_tup

    @property
    def trained_sfs(self):
        f = lambda x, y, z : tuple(
            (sfs.fit(x, y) for sfs in chain.from_iterable(self.sfs_tuple[z]))
        )

        lr_trained_sfs = f(self.genepool_df_loader.load_train_val_data.train_x,
                           self.genepool_df_loader.load_train_val_data.train_y,
                           0)
        dt_trained_sfs = f(self.genepool_df_loader.load_train_val_data.train_x,
                           self.genepool_df_loader.load_train_val_data.train_y,
                           1)

        lr_std_trained_sfs = f(self.genepool_df_loader.load_train_val_data.std_train_x,
                               self.genepool_df_loader.load_train_val_data.std_train_y,
                               2)
        dt_std_trained_sfs = f(self.genepool_df_loader.load_train_val_data.std_train_x,
                               self.genepool_df_loader.load_train_val_data.std_train_y,
                               3)
        return lr_trained_sfs, dt_trained_sfs, lr_std_trained_sfs, dt_std_trained_sfs

    @property
    def get_ranked_antibody(self):
        f = lambda x : [i.feature_names_in_[i.get_support()] for i in self.trained_sfs[x]]
        lr_antibody = f(0)
        dt_antibody = f(1)
        lr_std_antibody = f(2)
        dt_std_antibody = f(3)

        return lr_antibody, dt_antibody, lr_std_antibody, dt_std_antibody

    @property
    def sliced_df(self):
        def slice(tr_x, tr_y, val_x, val_y, gene_list):
            for gene in gene_list:
                yield tr_x[gene], tr_y, val_x[gene], val_y

        lr = slice(self.genepool_df_loader.load_train_val_data.train_x,
                   self.genepool_df_loader.load_train_val_data.train_y,
                   self.genepool_df_loader.load_train_val_data.val_x,
                   self.genepool_df_loader.load_train_val_data.val_y,
                   self.get_ranked_antibody[0])
        dt = slice(self.genepool_df_loader.load_train_val_data.train_x,
                   self.genepool_df_loader.load_train_val_data.train_y,
                   self.genepool_df_loader.load_train_val_data.val_x,
                   self.genepool_df_loader.load_train_val_data.val_y,
                   self.get_ranked_antibody[1])
        lr_std = slice(self.genepool_df_loader.load_train_val_data.std_train_x,
                       self.genepool_df_loader.load_train_val_data.std_train_y,
                       self.genepool_df_loader.load_train_val_data.std_val_x,
                       self.genepool_df_loader.load_train_val_data.std_val_y,
                       self.get_ranked_antibody[2])
        dt_std = slice(self.genepool_df_loader.load_train_val_data.std_train_x,
                       self.genepool_df_loader.load_train_val_data.std_train_y,
                       self.genepool_df_loader.load_train_val_data.std_val_x,
                       self.genepool_df_loader.load_train_val_data.std_val_y,
                       self.get_ranked_antibody[3])


        return lr, dt, lr_std, dt_std

    def get_performance(self, cm):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fdr = fp / (fp+tp)
        fpr = fp / (fp+tn)
        return accuracy, fdr, fpr

    def get_t_test_p_value(self, t_x, t_y, v_x, v_y):
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

    @property
    def prediction_performance(self):
        def pred(tup, estimator):

            t_x, t_y, v_x, v_y = tup
            train_m = estimator.fit(t_x, t_y)
            pred_train_x = train_m.predict(t_x)
            pred_val_x = train_m.predict(v_x)
            train_cm = confusion_matrix(t_y, pred_train_x)
            val_cm = confusion_matrix(v_y, pred_val_x)
            ttest_result = self.get_t_test_p_value(pred_train_x, t_y, pred_val_x, v_y)
            yield Genepool_rank_result_container(train_m, ttest_result, (self.get_performance(train_cm), self.get_performance(val_cm)))

        lr = partial(pred, estimator=self.lr_estimator)
        dt = partial(pred, estimator=self.dt_estimator)

        def f(t):
            l = map(lr, t[0])
            d = map(dt, t[1])
            ls = map(lr, t[2])
            ds = map(dt, t[3])
            return l, d, ls, ds

        return chain.from_iterable(f(self.sliced_df))

    @property
    def gene_pool_and_prediction_performance(self):
        gene_pool = repeat(self.genepool_df_loader,)
        pred_performance = tuple(self.prediction_performance)

        return zip(gene_pool, pred_performance)
