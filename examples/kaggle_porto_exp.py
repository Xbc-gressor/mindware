import pandas as pd
import os
import sys

# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import HPO
from mindware import EnsembleBuilder
from mindware import CLASSIFICATION
import pickle as pkl
import argparse
import numpy as np
from sklearn.metrics._scorer import _BaseScorer


def gini(actual, pred):
    assert len(actual) == len(pred)

    _all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    _all = _all[np.lexsort((_all[:, 2], -1 * _all[:, 1]))]
    total_losses = _all[:, 0].sum()
    gini_sum = _all[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.0

    return gini_sum / len(actual)


class GiniScorer(_BaseScorer):
    def __init__(self, **kwargs):
        super().__init__(score_func=gini, sign=1, kwargs=kwargs)

    def _score(self, method_caller, clf, X, y, sample_weight=None):

        y_pred = clf.predict_proba(X)[:, 1]
        return self._score_func(y, y_pred, **self._kwargs)


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
    evaluation = args.evaluation
    time_limit = args.time_limit
    per_time_limit = args.per_time_limit
    estimator_id = 'neural_network'

    # Load data
    # data_dir = 'D:\\xbc\Fighting\AutoML\datas\kaggle\porto-seguro-safe-driver-prediction'
    # data_dir = '/root/automl_data/kaggle/porto-seguro-safe-driver-prediction'
    data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/porto-seguro-safe-driver-prediction'

    dm = DataManager(na_values=[-1])

    cate_cols = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin',
                 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',
                 'ps_car_11_cat', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
                 'ps_calc_19_bin', 'ps_calc_20_bin']
    # train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), keep_default_na=True, header='infer', sep=',').astype(np.int64)
    # train_df[cate_cols] = train_df[cate_cols].astype(str)
    # train_df.replace('-1', np.nan, inplace=True)
    # train_data_node = dm.from_train_df(train_df, ignore_columns=['id'], label_name='target', cate_cols=cate_cols)

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['id'], label_name='target', cate_cols=cate_cols, na_values=[-1])
    train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=x_encode)

    # test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'), keep_default_na=True, header='infer', sep=',').astype(np.int64)
    # test_df[cate_cols] = test_df[cate_cols].astype(str)
    # test_df.replace('-1', np.nan, inplace=True)
    # test_data_node = dm.from_test_df(test_df, ignore_columns=['id'])

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['id'])
    test_data_node = dm.preprocess_transform(test_data_node)



    if Opt == 'cash':
        # 'lda',
        OPT = CASH
    else:
        OPT = CASHFE

    # hpo = OPT(
    #     include_algorithms=None, sub_optimizer='smac', task_type=task_type,
    #     metric=GiniScorer(),
    #     data_node=train_data_node, evaluation=evaluation, resampling_params=None,
    #     optimizer=optimizer, inner_iter_num_per_iter=5,
    #     time_limit=time_limit, amount_of_resource=100, per_run_time_limit=per_time_limit,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method=ensemble_method, ensemble_size=ensemble_size
    # )

    hpo = HPO(
        estimator_id=estimator_id, task_type=task_type,
        metric=GiniScorer(),
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer=optimizer,
        time_limit=time_limit, amount_of_resource=50, per_run_time_limit=per_time_limit,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=None, ensemble_size=ensemble_size
    )

    print(hpo.run())
    # pred_ens = hpo.predict(test_data_node, ens=True, prob=True)[:, 1]
    pred = hpo.predict(test_data_node, ens=False, prob=True)[:, 1]

    x_encode_str = '' if x_encode is None else ('_' + x_encode)

    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['id']
    result = pd.DataFrame({'id': passenger_id, 'target': pred})
    result.to_csv(os.path.join(data_dir, f'{Opt}{estimator_id}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result.csv'), index=False)
    print('Result has been saved to result.csv.')
    # result_ens = pd.DataFrame({'id': passenger_id, 'target': pred_ens})
    # result_ens.to_csv(os.path.join(data_dir, f'{Opt}{x_encode_str}_{evaluation}_{optimizer}{time_limit}_{ensemble_method}{ensemble_size}_result_ens.csv'), index=False)
    # print('Ensemble result has been saved to result_ens.csv.')

