import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--optimizer', type=str, default='smac', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default='normalize', help='normalize')
    parser.add_argument('--ensemble_method', type=str, default='blending', help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=5024, help='time limit')
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
    data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\titanic\\'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), label_name='Survived', ignore_columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    breakpoint()
    train_data_node = dm.preprocess_fit(train_data_node, x_encode=x_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_data_node = dm.preprocess_transform(test_data_node)

    breakpoint()
    # Initialize CASHFE

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
        data_node=train_data_node, evaluation=evaluation, resampling_params={'test_size': 0.25},
        optimizer=optimizer,
        time_limit=time_limit, amount_of_resource=100, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=None, ensemble_size=ensemble_size
    )

    print(opt_hpo.run())

    pred_hpo = opt_hpo.predict(test_data_node, ens=False)

    pred_hpo = dm.decode_label(pred_hpo)

    result_hpo = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_hpo})
    result_hpo.to_csv(os.path.join(data_dir,
                                   f'hpo{estimator_id}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_hpo.csv'),
                      index=False)

    opt = OPT(
        include_algorithms=None, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer=optimizer, inner_iter_num_per_iter=1,
        time_limit=time_limit, amount_of_resource=100, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=ensemble_size
    )

    print(opt.run())

    pred_ens = opt.predict(test_data_node, ens=True)
    pred = opt.predict(test_data_node, ens=False)
    pred_ens = dm.decode_label(pred_ens)
    pred = dm.decode_label(pred)

    result = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred})
    result.to_csv(os.path.join(data_dir,
                               f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result.csv'),
                  index=False)
    print('Result has been saved to result.csv.')

    result_ens = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_ens})
    result_ens.to_csv(os.path.join(data_dir,
                                   f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_ens.csv'),
                      index=False)
    print('Ensemble result has been saved to result_ens.csv.')


    breakpoint()