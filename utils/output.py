'''
save result yaml file outside function
'''
from ranker.get_antibody_ranking import get_antibody_ranking
from itertools import batched
import yaml
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression

def save_ranking_tuple_yaml(t:tuple, result_root:str)->None:
   result_list = []
   for lr, dt in t:
       for i, j, k, l in batched(lr,4):
           result_dict={}
           result_dict['divider'] = i
           result_dict['group'] = j
           result_dict['model'] = k
           rank_list = [(int(rank), ab) for rank, ab in l]
           rank_list_ = sorted(rank_list, key=lambda x: x[0], reverse=False)
           result_dict['rank_list'] = rank_list_
           result_list.append(result_dict)
       for i, j, k, l in batched(dt, 4):
           result_dict={}
           result_dict['divider'] = i
           result_dict['group'] = j
           result_dict['model'] = k
           rank_list = [(int(rank), ab) for rank, ab in l]
           rank_list_ = sorted(rank_list, key=lambda x: x[0], reverse=False)
           result_dict['rank_list'] = rank_list_
           result_list.append(result_dict)
   with open(result_root, 'w') as f:
       yaml.dump(result_list, f, default_flow_style=False)

def save_result(result_dict_list:list[dict], result_root:str)->None:
    if len(result_dict_list) == 0:
        print('You have no result.')
    else:
        with open(result_root, 'w') as f:
            yaml.dump(result_dict_list, f, default_flow_style=False)


def get_model_result(estimator:object)->dict:
    result_dict = {}
    if isinstance(estimator, DecisionTreeClassifier):
        result_dict['Model'] = str(estimator)
        # result_dict['Param'] = estimator.get_params()
        # try: result_dict['Dot_data'] = export_graphviz(estimator,)
        # except: result_dict['Dot_data'] = 'Not fitted '
    elif isinstance(estimator, LogisticRegression):
        result_dict['Model'] = str(estimator)
        result_dict['Param'] = estimator.get_params()
    else: pass

    return result_dict





def formatting_performance_and_model(t:tuple, scaled=None)->list[dict]:
    if scaled is None:
        scale = 'non_scaled'
    else: scale = 'scaled'
    if len(t) == 0:
        return None
    else:
        result_dict_list=[]
        for i in t:
            container = i[0].container
            result_container = i[1]
            result_dict = {}
            result_dict['Feature_selection_model'] = get_model_result(container.model)
            result_dict['Antibody'] = tuple(result_container.antibody)
            result_dict['Classification_model'] = get_model_result(result_container.base_model)
            train_cm, val_cm = result_container.cm
            train_accuracy, train_fdr, train_fpr = train_cm
            val_accuracy, val_fdr, val_fpr = val_cm
            p_value_btw_Rs, p_value_btw_NRs, p_value_in_train, p_value_in_val = result_container.ttest_p_value
            result_dict['Train_accuracy'] = float(train_accuracy)
            result_dict['Train_fdr'] = float(train_fdr)
            result_dict['Train_fpr'] = float(train_fpr)
            result_dict['Validation_accuracy'] = float(val_accuracy)
            result_dict['Validation_fdr'] = float(val_fdr)
            result_dict['Validation_fpr'] = float(val_fpr)
            result_dict['p_value_btw_Rs'] = float(p_value_btw_Rs)
            result_dict['p_value_btw_NRs'] = float(p_value_btw_NRs)
            result_dict['p_value_in_train'] = float(p_value_in_train)
            result_dict['p_value_in_val'] = float(p_value_in_val)
            if scale == 'non_scaled':
                result_dict['Gene_pool'] = tuple(container.non_scaled_pool)
            else:
                result_dict['Gene_pool'] = tuple(container.std_pool)

            result_dict_list.append(result_dict)

    return result_dict_list







if __name__ == '__main__':
    save_ranking_tuple_yaml(get_antibody_ranking('../config/config.yaml'), '../src/ranking_list.yaml')
