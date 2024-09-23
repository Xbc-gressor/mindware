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
import pickle as pkl

if __name__ == '__main__':
    task_type = REGRESSION
    # Load data
    data_dir = 'D:\\Code\\MindWare\\Data\\bike-sharing-demand'
    # data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/houseprice'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['datetime'],
                                        label_name='count')
    # breakpoint()
    train_data_node = dm.preprocess_fit(train_data_node, task_type)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['datetime'])
    test_data_node = dm.preprocess_transform(test_data_node)

    # Initialize CASHFE

    ensemble_method = "ensemble_selection"
    ensemble_size = 5
    metric = 'rmse'
    evaluation = 'holdout'

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'lasso_regression', 'liblinear_svr', 'libsvm_svr',
        'random_forest', 'ridge_regression'
    ]

    # 'lda',
    # hpo = CASH(
    #     include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
    #     metric=metric,
    #     data_node=train_data_node, evaluation=evaluation, resampling_params=None,
    #     optimizer='smac', inner_iter_num_per_iter=1,
    #     time_limit=1024, amount_of_resource=100, per_run_time_limit=600,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method=ensemble_method, ensemble_size=ensemble_size
    # )

    hpo = CASHFE(
        include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer='mab', inner_iter_num_per_iter=5,
        time_limit=3024, amount_of_resource=100, per_run_time_limit=600,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=5
    )

    print(hpo.run())
    hpo.refit()
    pred_ens = hpo.predict(test_data_node, ens=True)
    pred = hpo.predict(test_data_node, ens=False)

    pred = dm.decode_label(pred)
    pred_ens = dm.decode_label(pred_ens)

    date = pd.read_csv(os.path.join(data_dir, 'test.csv'))['datetime']
    result = pd.DataFrame({'datetime': date, 'count': pred})
    result.to_csv(os.path.join(data_dir, 'cashfe_mab3024_sel5_result.csv'), index=False)
    print('Result has been saved to result.csv.')
    result_ens = pd.DataFrame({'datetime': date, 'count': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, 'cashfe_mab3024_sel5_result_ens.csv'), index=False)
    print('Ensemble result has been saved to result_ens.csv.')

    # config_path = 'D:\\xbc\\Fighting\\AutoML\\mindware\\examples\\data\\CASH-smac(1)_2024-06-04-21-21-08-961071\\2024-06-04-21-21-08-961071_topk_config.pkl'
    # with open(config_path, 'rb') as f:
    #     stats = pkl.load(f)
    #
    # # Ensembling all intermediate/ultimate models found in above optimization process.
    # es = EnsembleBuilder(stats=stats,
    #                      data_node=train_data_node,
    #                      ensemble_method=ensemble_method,
    #                      ensemble_size=ensemble_size*2,
    #                      task_type=task_type,
    #                      metric=hpo.metric,
    #                      output_dir=hpo.output_dir)
    #
    # es.fit(train_data_node)
    # breakpoint()
