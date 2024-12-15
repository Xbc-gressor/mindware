import pandas as pd
import os
import sys


# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.utils.data_manager import DataManager
from mindware import CASH
from mindware import REGRESSION

if __name__ == '__main__':

    x_encode = None
    label_encode = None
    ensemble_method = "ensemble_selection"
    ensemble_size = 5
    metric = 'msle'
    evaluation = 'cv'
    estimator_id = 'gradient_boosting'

    task_type = REGRESSION


    # Load data
    # data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\houseprices\\'
    data_dir = '/Users/xubeideng/Documents/icloud/Scientific Research/AutoML/automl_data/kaggle/houseprice'
    # data_dir = '/root/automl_data/kaggle/houseprice'

    dm = DataManager()

    train_data_node_ori = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['Id'],
                                        label_name='SalePrice')
    train_data_node = dm.preprocess_fit(train_data_node_ori, task_type, x_encode=x_encode, label_encode=label_encode)

    test_data_node_ori = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['Id'])
    test_data_node = dm.preprocess_transform(test_data_node_ori)

    breakpoint()

