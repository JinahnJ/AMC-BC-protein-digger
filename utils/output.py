'''
save result yaml file outside function
'''
from ranker.get_antibody_ranking import get_antibody_ranking
from itertools import batched
import yaml

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

if __name__ == '__main__':
    save_ranking_tuple_yaml(get_antibody_ranking('../config/config.yaml'), '../src/ranking_list.yaml')
