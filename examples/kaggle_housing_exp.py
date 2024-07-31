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

    x_encode = 'normalize'
    label_encode = 'normalize'
    ensemble_method = "ensemble_selection"
    ensemble_size = 5
    metric = 'rmse'
    evaluation = 'holdout'
    estimator_id = 'neural_network'

    task_type = REGRESSION


    # Load data
    data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\houseprices\\'
    # data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/houseprice'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['Id'],
                                        label_name='SalePrice')
    train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=x_encode, label_encode=label_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['Id'])
    test_data_node = dm.preprocess_transform(test_data_node)

    breakpoint()

    # Initialize CASHFE

    x_encode_str = '' if x_encode is None else ('_' + x_encode)
    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['Id']




    opt_hpo = HPO(
        estimator_id='neural_network', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params={'test_size': 0.25},
        optimizer='smac',
        time_limit=3024, amount_of_resource=10, per_run_time_limit=600,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=None, ensemble_size=ensemble_size
    )

    print(opt_hpo.run())
    pred_hpo = opt_hpo.predict(test_data_node, ens=False)

    pred_hpo = dm.decode_label(pred_hpo)

    result_hpo = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred_hpo})
    result_hpo.to_csv(os.path.join(data_dir,
                               f'hpo{estimator_id}{x_encode}{label_encode}_{evaluation}_result.csv'),
                  index=False)
    print('Result has been saved to result_hpo.csv.')



    opt = CASHFE(
        include_algorithms=None, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer='smac', inner_iter_num_per_iter=1,
        time_limit=3024, amount_of_resource=10, per_run_time_limit=600,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=5
    )

    print(opt.run())

    pred_ens = opt.predict(test_data_node, ens=True)
    pred = opt.predict(test_data_node, ens=False)

    pred = dm.decode_label(pred)
    pred_ens = dm.decode_label(pred_ens)

    result = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred})
    result.to_csv(os.path.join(data_dir, f'cashfe{x_encode}{label_encode}_mab3024_sel5_result.csv'), index=False)
    print('Result has been saved to result.csv.')
    result_ens = pd.DataFrame({'Id': passenger_id, 'SalePrice': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, f'cashfe{x_encode}{label_encode}_mab3024_sel5_result_ens.csv'), index=False)
    print('Ensemble result has been saved to result_ens.csv.')



    breakpoint()
