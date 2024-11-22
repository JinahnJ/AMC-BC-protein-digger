from dataclasses import dataclass
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import scale
import numpy as np

@dataclass(frozen=True)
class Paired_dataframe:
    name: str
    df_group_tuple: tuple #composed of pd.df.groupby object
    value_name:tuple=('Antibody', 'Value')
    y_value_name:str='response',
    target_column:str='Antibody'
    x_value_name:str='Value'

    @property
    def key_df_tup(self):
       tup = tuple((key, group) for key, group in self.df_group_tuple)
       return tup

    @staticmethod
    def pivot_df(df: pd.DataFrame, target_column='Antibody', x_value='Value', y_value='response', sn_column='SN'):
        df = df.copy()
        df['id'] = df.groupby(target_column, observed=False).cumcount()
        df_x = df.pivot(index='id', columns=target_column, values=x_value)
        df_y = df.pivot(index='id', columns=target_column, values=y_value)
        sr_y = df_y.iloc[:,0]
        sr_sn = df.pivot(index='id', columns=target_column, values=sn_column).iloc[:, 0]  # SN Series

        return tuple([df_x, sr_y, sr_sn])

    @property
    def pivoted_df(self):
        tup = self.key_df_tup
        group_tup = ((key, df) for key, df in tup)
        group_tup_xy = ((key, self.pivot_df(df)) for key, df in group_tup)
        tup_ = ((self.name, group_name, df_x, df_y) for group_name, (df_x, df_y, _) in group_tup_xy)
        return tup_


#Paired dataframe ->  train, RFECV result
@dataclass(frozen=True)
class RFECV_result:
    divider_name : tuple
    group_name: tuple
    x :  pd.DataFrame
    y : pd.DataFrame
    model : object
    scale : object= scale
    kwargs : dict =None #**kwargs for RFECV setting

    @property
    def make_RFECV(self):
        if self.kwargs is not None:
            selector = RFECV(estimator=self.model, min_features_to_select=4, **self.kwargs)
            return selector
        else:
            selector = RFECV(estimator=self.model, min_features_to_select=4)
            return selector

    @property
    def train_selector(self):
        trained_selector = self.make_RFECV.fit(self.x, self.y)
        return trained_selector

    @property
    def rank_list(self):
        fixed_ranking = np.copy(self.train_selector.ranking_)
        ranking_list = zip(map(int, fixed_ranking), self.x.columns)
        return tuple(ranking_list)

    @staticmethod
    def slicing_rank_list(ranking_list:zip, n:int):
       tup = tuple((rank, antibodyname) for rank, antibodyname in ranking_list if int(rank) < n+1)
       return tup

    @property
    def std_scaled_df(self):
        std_x = pd.DataFrame(self.scale(self.x), columns=self.x.columns)
        return std_x

    @property
    def make_std_RFECV(self):
        if self.kwargs is not None:
            selector = RFECV(estimator=self.model, min_features_to_select=4, **self.kwargs)
            return selector
        else:
            selector = RFECV(estimator=self.model, min_features_to_select=4)
            return selector

    @property
    def std_train_selector(self):
        trained_selector = self.make_std_RFECV.fit(self.std_scaled_df, self.y)
        return trained_selector

    @property
    def std_rank_list(self):
        fixed_ranking = np.copy(self.std_train_selector.ranking_)
        ranking_list = zip(map(int, fixed_ranking), self.x.columns)
        return tuple(ranking_list)


