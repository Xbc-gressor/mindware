import os
import pandas as pd


def get_dataset_info(task_type, dataset):
    assert task_type in ['CLS', 'RGS']
    datasets_dir = '/root/automl_data/automl_data/'
    # if task_type == 'RGS':
    #     datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='REG_SORT')
    # else:
    #     datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='CLS')

    if task_type == 'RGS':
        dataset_path = os.path.join(datasets_dir, 'rgs_datasets', dataset + '.csv')
    else:
        dataset_path = os.path.join(datasets_dir, 'cls_datasets', dataset + '.csv')

    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps', 'vehicle_sensIT', 'codrna']:
        label_column = 0
    elif dataset in ['rmftsa_sleepdata(1)']:
        label_column = 1
    else:
        label_column = -1

    if dataset in ['spambase', 'messidor_features']:
        header = None
    else:
        header = 'infer'

    if dataset in ['winequality_white', 'winequality_red']:
        sep = ';'
    else:
        sep = ','

    return dataset_path, label_column, header, sep