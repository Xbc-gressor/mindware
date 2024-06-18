import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from mindware.utils.data_manager import DataManager
from mindware.modules.cash.base_cash import BaseCASH

from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *
from mindware import candidates_classifiers

# if __name__ == '__main__':
#     iris = load_iris()
#     X, y = iris.data, iris.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#     dm = DataManager(X_train, y_train)
#     train_data = dm.get_data_node(X_train, y_train)
#     test_data = dm.get_data_node(X_test, y_test)
#
#     metric = 'acc'
#     scorer = get_metric(metric)
#     resampling_strategy = 'holdout'
#
#     include_algorithms = None
#     # 'lda',
#     hpo = BaseCASH(
#         include_algorithms=include_algorithms, sub_optimizer='smac',
#         metric=metric,
#         data_node=train_data, evaluation='holdout', resampling_params=None,
#         optimizer='mab', per_run_time_limit=600,
#         time_limit=2024, amount_of_resource=100,
#         inner_iter_num_per_iter=3,
#         output_dir='./data', seed=1, n_jobs=1,
#         ensemble_method="ensemble_selection", ensemble_size=10
#     )
#     print(hpo.run())
#     y_true = test_data.data[1]
#     pred_ens = hpo.predict(test_data, ens=True)
#     pred = hpo.predict(test_data, ens=False)
#
#     ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
#     pred_perf = scorer._score_func(y_true, pred) * scorer._sign
#
#     print('ens_perf: %f' % ens_perf)
#     print('pred_perf: %f' % pred_perf)
#     # ens_perf: 0.980000
#     # pred_perf: 0.960000
#
#     hpo.refit()
#     y_true = test_data.data[1]
#     pred_ens = hpo.predict(test_data, ens=True)
#     pred = hpo.predict(test_data, ens=False)
#
#     ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
#     pred_perf = scorer._score_func(y_true, pred) * scorer._sign
#
#     print('ens_perf: %f' % ens_perf)
#     print('pred_perf: %f' % pred_perf)
#
#     # ens_perf: 0.980000
#     # pred_perf: 0.960000
#     breakpoint()


if __name__ == '__main__':
    housing = fetch_california_housing()

    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    metric = 'mse'
    scorer = get_metric(metric)
    resampling_strategy = 'holdout'

    include_algoriths = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'k_nearest_neighbors', 'lasso_regression', 'liblinear_svr', 'libsvm_svr',
        'random_forest', 'ridge_regression', 'lightgbm'
    ]
    hpo = BaseCASH(
        include_algorithms=None,
        metric=metric, sub_optimizer='smac',
        data_node=train_data, evaluation='holdout', resampling_params=None,
        optimizer='mab', per_run_time_limit=600,
        time_limit=2024, amount_of_resource=100,
        inner_iter_num_per_iter=3,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method="blending", ensemble_size=10,
    )

    print(hpo.run())

    breakpoint()
    y_true = test_data.data[1]
    pred_ens = hpo.predict(test_data, ens=True)
    pred = hpo.predict(test_data, ens=False)

    ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
    pred_perf = scorer._score_func(y_true, pred) * scorer._sign

    print('ens_perf: %f' % ens_perf)
    print('pred_perf: %f' % pred_perf)

    # mab 1: -0.412963, -0.413910
    # mab 3: -0.345184, -0.429771
    # smac : -0.420530, -0.423739

    hpo.refit()
    y_true = test_data.data[1]
    pred_ens = hpo.predict(test_data, ens=True)
    pred = hpo.predict(test_data, ens=False)

    ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
    pred_perf = scorer._score_func(y_true, pred) * scorer._sign

    print('ens_perf: %f' % ens_perf)
    print('pred_perf: %f' % pred_perf)

    breakpoint()


# -0.20955691016965125 pred_perf: -0.208123 ens_perf: -0.214844
