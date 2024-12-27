import pandas as pd
import os
import sys


# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import EnsembleBuilder
from mindware import REGRESSION
from mindware import HPO
import pickle as pkl

if __name__ == '__main__':

    x_encode = None
    label_encode = None
    ensemble_method = "ensemble_selection"
    ensemble_size = 50
    metric = 'msle'
    evaluation = 'cv'
    estimator_id = 'gradient_boosting'
    time_limit = 7200

    task_type = REGRESSION


    # Load data
    # data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\houseprices\\'
    # data_dir = '/Users/xubeideng/Documents/icloud/Scientific Research/AutoML/automl_data/kaggle/houseprice'
    data_dir = '/root/automl_data/kaggle/houseprice'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), drop_index=[0], label_col=-1)
    train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=x_encode, label_encode=label_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'))
    test_data_node = dm.preprocess_transform(test_data_node)


    # Initialize CASHFE

    x_encode_str = '' if x_encode is None else ('_' + x_encode)
    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['Id']

    # opt_hpo = HPO(
    #     estimator_id='gradient_boosting', task_type=task_type,
    #     metric=metric,
    #     data_node=train_data_node, evaluation=evaluation,
    #     optimizer='smac',
    #     time_limit=3024, amount_of_resource=100, per_run_time_limit=600,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method=ensemble_method, ensemble_size=ensemble_size
    # )
    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'random_forest', 'lightgbm', 'xgboost'
    ]
    opt = CASHFE(
        include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer='block_1', inner_iter_num_per_iter=10,
        time_limit=time_limit, amount_of_resource=int(1e6), per_run_time_limit=300,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=ensemble_size
    )

    print(opt.get_conf(save=True))  # 保存设置
    print(opt.run())
    print(opt.get_model_info(save=True))  # 保存最优模型信息
    pred_hpo = opt.predict(test_data_node, ens=False)

    pred_hpo = dm.decode_label(pred_hpo)

    result_hpo = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred_hpo})
    result_hpo.to_csv(os.path.join(data_dir,
                               f'Block_1{time_limit}{ensemble_method}{ensemble_size}_{evaluation}_result.csv'),
                  index=False)
    print('Result has been saved to result_hpo.csv.')
    
    
    pred_ens = opt.predict(test_data_node, ens=True)
    pred_ens = dm.decode_label(pred_ens)
    result_ens = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred_ens})
    result_ens.to_csv(os.path.join(data_dir,
                               f'Block_1{time_limit}{ensemble_method}{ensemble_size}_{evaluation}_result_ens.csv'),
                  index=False)
    print('Ensemble result has been saved to result_ens.csv.')



    # opt = CASHFE(
    #     include_algorithms=['gradient_boosting'], sub_optimizer='smac', task_type=task_type,
    #     metric=metric,
    #     data_node=train_data_node, evaluation=evaluation, resampling_params=None,
    #     optimizer='smac', inner_iter_num_per_iter=1,
    #     time_limit=3600, amount_of_resource=10000, per_run_time_limit=60,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method=ensemble_method, ensemble_size=ensemble_size
    # )

    # print(opt.run())

    # pred_ens = opt.predict(test_data_node, ens=True)
    # pred = opt.predict(test_data_node, ens=False)

    # pred = dm.decode_label(pred)
    # pred_ens = dm.decode_label(pred_ens)

    # result = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred})
    # result.to_csv(os.path.join(data_dir, f'cashfe{x_encode}{label_encode}_mab3024_sel5_result.csv'), index=False)
    # print('Result has been saved to result.csv.')
    # result_ens = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred_ens})
    # result_ens.to_csv(os.path.join(data_dir, f'cashfe{x_encode}{label_encode}_mab3024_sel5_result_ens.csv'), index=False)
    # print('Ensemble result has been saved to result_ens.csv.')



    breakpoint()
