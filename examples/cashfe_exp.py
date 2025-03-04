import argparse
import os
import sys
import time

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindware.utils.data_manager import DataManager
from mindware.modules.cashfe.base_cashfe import BaseCASHFE

from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *

if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    metric = 'acc'
    scorer = get_metric(metric)
    resampling_strategy = 'holdout'

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'xgboost', 'random_forest', 'lightgbm'
    ]
    # 'lda',
    hpo = BaseCASHFE(
        include_algorithms=include_algorithms, sub_optimizer='smac',
        metric=metric,
        data_node=train_data, evaluation='holdout', resampling_params=None,
        optimizer='block_1', per_run_time_limit=600,
        time_limit=1024, amount_of_resource=60,
        inner_iter_num_per_iter = 10,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method="blending", ensemble_size=5
    )
    print(hpo.run())
    y_true = test_data.data[1]
    pred_ens = hpo.predict(test_data, ens=True)
    pred = hpo.predict(test_data, ens=False)

    ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
    pred_perf = scorer._score_func(y_true, pred) * scorer._sign

    breakpoint()





# if __name__ == '__main__':
#     housing = fetch_california_housing()
#
#     X, y = housing.data[:100], housing.target[:100]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#     dm = DataManager(X_train, y_train)
#     train_data = dm.get_data_node(X_train, y_train)
#     test_data = dm.get_data_node(X_test, y_test)
#
#     metric = 'mse'
#     scorer = get_metric(metric)
#     resampling_strategy = 'holdout'
#
#     include_algoriths = [
#         'adaboost', 'extra_trees', 'gradient_boosting',
#         'k_nearest_neighbors', 'lasso_regression', 'liblinear_svr', 'libsvm_svr',
#         'random_forest', 'ridge_regression'
#     ]
#     hpo = BaseCASH(
#         metric=metric,
#         data_node=train_data, evaluation='holdout', resampling_params=None,
#         optimizer='smac', per_run_time_limit=600,
#         time_limit=1024, amount_of_resource=10,
#         output_dir='./data', seed=1, n_jobs=1,
#         ensemble_method="blending", ensemble_size=5,
#         include_algorithms=['lightgbm']
#     )
#
#     print(hpo.run())
#     y_true = test_data.data[1]
#     pred_ens = hpo.predict(test_data, ens=True)
#     pred = hpo.predict(test_data, ens=False)
#
#     ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
#     pred_perf = scorer._score_func(y_true, pred) * scorer._sign
#
#     breakpoint()
