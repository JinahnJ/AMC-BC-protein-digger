from utils.dataframe_utils import config_file, generate_pair_tuple, get_dataframe, generate_pair_dataframe
from utils.paired_df_dcls import Paired_dataframe, RFECV_result
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # c = './config/config.yaml'
    # config_dir = lambda x: config_file(x)
    # target_tuple_pair = lambda root, x='target_category': generate_pair_tuple(config_dir(root)[x])
    # dataset_df = lambda root, x='dataset_file' : get_dataframe(config_dir(root)[x])
    # paired_dataframe = lambda t, df, d_cls=Paired_dataframe : generate_pair_dataframe(t, df, d_cls=d_cls)
    # model = LogisticRegression(class_weight='balanced', max_iter=10, solver='liblinear')

    def main(c=c, model=model):
        # print(config_dir(c))
        # print(target_tuple_pair(c))
        # print(dataset_df(c))
        # print(paired_dataframe(target_tuple_pair(c), dataset_df(c), d_cls=Paired_dataframe))
        # for i in paired_dataframe(target_tuple_pair(c), dataset_df(c), d_cls=Paired_dataframe):
        #     for j, k, l, m  in i.pivoted_df:
        #         print('name:', j)
        #         print('group_name:', k)
        #         print('df_x', l.head(10))
        #         print('df_y', m.head(10))
        #         print('-'*15)

        #TODO: RFECV test
        # for i in paired_dataframe(target_tuple_pair(c), dataset_df(c), d_cls=Paired_dataframe):
        #    for j, k, l in i.pivoted_df:
        #        m = RFECV_result(j, k, l, model)
        #        print(m.y)





        pass

    main()

