
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import CLASSIFICATION
from mindware import EnsembleBuilder
import pickle as pkl

if __name__ == '__main__':

    # Load data
    data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/spaceship'

    # 预处理数据，将train和test表格中 Cabin 一列形如 B/0/P 的数据中的第一个和最后一个字母提取出来，形成两列 deck 和 side，并保存
    # train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    # train_data['deck'] = train_data['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)
    # train_data['side'] = train_data['Cabin'].apply(lambda x: x[-1] if pd.notnull(x) else np.nan)
    # train_data.to_csv(os.path.join(data_dir, 'train_new.csv'), index=False)
    #
    # test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # test_data['deck'] = test_data['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)
    # test_data['side'] = test_data['Cabin'].apply(lambda x: x[-1] if pd.notnull(x) else np.nan)
    # test_data.to_csv(os.path.join(data_dir, 'test_new.csv'), index=False)

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train_new.csv'), label_name='Transported', ignore_columns=['PassengerId', 'Name', 'Cabin'])
    train_data_node = dm.preprocess_fit(train_data_node, CLASSIFICATION)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test_new.csv'), ignore_columns=['PassengerId', 'Name', 'Cabin'])
    test_data_node = dm.preprocess_transform(test_data_node)

    # Initialize CASHFE

    metric = 'acc'
    evaluation = 'holdout'
    ensemble_method = "blending"
    ensemble_size = 10
    task_type = CLASSIFICATION

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'liblinear_svc', 'libsvm_svc',
        'logistic_regression', 'qda', 'random_forest',
        'lightgbm'
    ]

    # 'lda',
    hpo = CASHFE(
        include_algorithms=include_algorithms, sub_optimizer='smac', task_type=CLASSIFICATION,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer='mab', inner_iter_num_per_iter=5,
        time_limit=2024, amount_of_resource=100, per_run_time_limit=600,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=ensemble_size
    )

    # hpo = BaseCASH(
    #     include_algorithms=include_algorithms, sub_optimizer='smac',
    #     metric=metric,
    #     data_node=train_data_node, evaluation='holdout', resampling_params=None,
    #     optimizer='mab', per_run_time_limit=600,
    #     time_limit=1024, amount_of_resource=100,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method="blending", ensemble_size=5
    # )

    # print(hpo.run())
    # hpo.refit()
    # pred_ens = hpo.predict(test_data_node, ens=True)
    # pred = hpo.predict(test_data_node, ens=False)
    #
    # pred_ens = dm.decode_label(pred_ens)
    # pred = dm.decode_label(pred)
    #
    # passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['PassengerId']
    # result = pd.DataFrame({'PassengerId': passenger_id, 'Transported': pred})
    # result.to_csv(os.path.join(data_dir, 'cashfe_mab2024_ble10_result.csv'), index=False)
    # print('Result has been saved to result.csv.')
    # result_ens = pd.DataFrame({'PassengerId': passenger_id, 'Transported': pred_ens})
    # result_ens.to_csv(os.path.join(data_dir, 'cashfe_mab2024_ble10_result_ens.csv'), index=False)
    # print('Ensemble result has been saved to result_ens.csv.')

    config_path = '/Users/xubeideng/Documents/Scientific Research/AutoML/code/mindware/examples/data/CASHFE-mab(1)_2024-06-05-16-40-50-634829/2024-06-05-16-40-50-634829_topk_config.pkl'
    with open(config_path, 'rb') as f:
        stats = pkl.load(f)

    breakpoint()