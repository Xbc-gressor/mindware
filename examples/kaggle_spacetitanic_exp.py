
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import HPO
from mindware import CLASSIFICATION, candidates_classifiers
from mindware import EnsembleBuilder
import pickle as pkl

import argparse

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS     # export NUMEXPR_NUM_THREADS=1

if __name__ == '__main__':

    # 从命令行参数中解析出参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--optimizer', type=str, default='smac', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default='normalize', help='normalize')
    parser.add_argument('--ensemble_method', type=str, default='blending', help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=2024, help='time limit')
    parser.add_argument('--per_time_limit', type=int, default=300, help='time limit')
    args = parser.parse_args()


    Opt = args.Opt

    task_type = CLASSIFICATION

    optimizer = args.optimizer
    x_encode = args.x_encode
    ensemble_method = args.ensemble_method
    ensemble_size = args.ensemble_size
    metric = 'acc'
    evaluation = args.evaluation
    time_limit = args.time_limit
    per_time_limit = args.per_time_limit
    estimator_id = 'neural_network'

    # Load data
    # data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/spaceship'
    data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\spaceship'
    # data_dir = '/root/automl_data/kaggle/spaceship'

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
    train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=x_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test_new.csv'), ignore_columns=['PassengerId', 'Name', 'Cabin'])
    test_data_node = dm.preprocess_transform(test_data_node)
    # Initialize CASHFE

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'lasso_regression', 'liblinear_svr', 'libsvm_svr',
        'random_forest', 'ridge_regression'
    ]

    if Opt == 'cash':
        # 'lda',
        OPT = CASH
    elif Opt == 'hpo':
        OPT = HPO
    else:
        OPT = CASHFE

    x_encode_str = '' if x_encode is None else ('_'+x_encode)
    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['PassengerId']

    opt_hpo = HPO(
        estimator_id='neural_network', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params={'test_size': 0.2},
        optimizer=optimizer,
        time_limit=time_limit, amount_of_resource=50, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=None, ensemble_size=ensemble_size
    )

    print(opt_hpo.run())

    pred_hpo = opt_hpo.predict(test_data_node, ens=False)

    pred_hpo = dm.decode_label(pred_hpo)

    result_hpo = pd.DataFrame({'PassengerId': passenger_id, 'Transported': pred_hpo})
    result_hpo.to_csv(os.path.join(data_dir,
                                   f'hpo{estimator_id}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_hpo.csv'),
                      index=False)



    opt = OPT(
        include_algorithms=None, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer=optimizer, inner_iter_num_per_iter=1,
        time_limit=time_limit, amount_of_resource=50, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=ensemble_size
    )

    print(opt.run())
    
    pred_ens = opt.predict(test_data_node, ens=True)
    pred = opt.predict(test_data_node, ens=False)
    pred_ens = dm.decode_label(pred_ens)
    pred = dm.decode_label(pred)

    result = pd.DataFrame({'PassengerId': passenger_id, 'Transported': pred})
    result.to_csv(os.path.join(data_dir, f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result.csv'), index=False)
    print('Result has been saved to result.csv.')

    result_ens = pd.DataFrame({'PassengerId': passenger_id, 'Transported': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_ens.csv'), index=False)
    print('Ensemble result has been saved to result_ens.csv.')




    breakpoint()
