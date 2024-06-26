import pandas as pd
import os
import sys


# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import EnsembleBuilder
from mindware import CLASSIFICATION
import pickle as pkl
import argparse

if __name__ == '__main__':

    # 从命令行参数中解析出参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--optimizer', type=str, default='smac', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default=None, help='smac or mab')
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

    # Load data
    data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\santander'
    # data_dir = '/root/automl_data/kaggle/santander'

    dm = DataManager()

    _train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['ID_code'],
                                        label_name='target')
    train_data_node = dm.preprocess_fit(_train_data_node, task_type, x_encode=x_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['ID_code'])
    test_data_node = dm.preprocess_transform(test_data_node)

    breakpoint()

    # Initialize CASHFE

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'lasso_regression', 'liblinear_svr', 'libsvm_svr',
        'random_forest', 'ridge_regression'
    ]

    if Opt == 'cash':
        # 'lda',
        OPT = CASH
    else:
        OPT = CASHFE

    hpo = OPT(
        include_algorithms=None, sub_optimizer='smac', task_type=task_type,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer=optimizer, inner_iter_num_per_iter=5,
        time_limit=time_limit, amount_of_resource=100, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=ensemble_size
    )

    print(hpo.run())
    pred_ens = hpo.predict(test_data_node, ens=True)
    pred = hpo.predict(test_data_node, ens=False)

    pred = dm.decode_label(pred)
    pred_ens = dm.decode_label(pred_ens)

    x_encode_str = '' if x_encode is None else ('_' + x_encode)

    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['ID_code']
    result = pd.DataFrame({'Id_code': passenger_id, 'target': pred})
    result.to_csv(os.path.join(data_dir, f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result.csv'), index=False)
    print('Result has been saved to result.csv.')
    result_ens = pd.DataFrame({'Id_code': passenger_id, 'target': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_ens.csv'), index=False)
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
