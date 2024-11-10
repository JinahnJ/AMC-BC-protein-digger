from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from utils.input import config_file

def LR_model(config_dict_root:str, random_state=None):
    config_dict = config_file(config_dict_root)
    lr_model_setting_dict = config_dict.get('model_setting_LogisticRegression')
    if random_state is not None:
        lr_model_setting_dict['random_state'] = random_state
    lr_model = LogisticRegression(**lr_model_setting_dict)
    return lr_model

def DT_model(config_dict_root:str, random_state=None):
    config_dict = config_file(config_dict_root)
    dt_model_setting_dict = config_dict.get('model_setting_DecisionTree')
    if random_state is not None:
        dt_model_setting_dict['random_state'] = random_state
    dt_model = DecisionTreeClassifier(**dt_model_setting_dict)
    return dt_model

def SFS_setting(config_dict_root:str):
    config_dict = config_file(config_dict_root)
    sfs_model_setting_dict = config_dict.get('model_setting_SequentialFeatureSelector')
    return sfs_model_setting_dict