import yaml

'''
read configuration file 
'''
def config_file(root='./config/config.yaml'):
    with open(root, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

def get_threshold(config_dict:dict)->dict:
    threshold_dict = config_dict['threshold']
    return threshold_dict

