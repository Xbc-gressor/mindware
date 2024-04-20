import argparse
import os
import sys
import time

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from mindware.utils.data_manager import DataManager
from mindware.modules.fe.base_fe import BaseFE

from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *


# if __name__ == '__main__':
#     iris = load_iris()
#     X, y = iris.data, iris.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#     dm = DataManager(X_train, y_train)
#     train_data = dm.get_data_node(X_train, y_train)
#     test_data = dm.get_data_node(X_test, y_test)
#
#     estimator_id = 'lightgbm'
#     metric = 'acc'
#     scorer = get_metric(metric)
#     resampling_strategy = 'holdout'
#
#     # timestamp = time.time()
#     # clf_class = _classifiers[estimator_id]
#     # cs = get_task_hyperparameter_space(CLASSIFICATION)
#     # fixed_config = clf_class.get_hyperparameter_search_space().get_default_configuration()
#     # config = cs.sample_configuration()
#     # hpo_eva = FEClassificationEvaluator(estimator_id=estimator_id,
#     #                                      fixed_config=fixed_config,
#     #                                      scorer=scorer,
#     #                                      data_node=train_data,
#     #                                      resampling_strategy=resampling_strategy,
#     #                                      timestamp=timestamp,
#     #                                      output_dir='./data',
#     #                                      seed=1,
#     #                                      if_imbal=False)
#     #
#     # print(hpo_eva(config))
#     hpo = BaseFE(estimator_id=estimator_id,
#                  metric=metric,
#                  data_node=train_data, evaluation='cv', resampling_params=None,
#                  optimizer='smac', per_run_time_limit=600,
#                  time_limit=1024, amount_of_resource=5,
#                  output_dir='./data', seed=1, n_jobs=1,
#                  ensemble_method="blending", ensemble_size=5)
#     # smac, random_search, tpe, partial_bohb
#
#     print(hpo.run())
#     breakpoint()
#     y_true = test_data.data[1]
#     pred_ens = hpo.predict(test_data, ens=True)
#     pred = hpo.predict(test_data, ens=False)
#
#     ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
#     pred_perf = scorer._score_func(y_true, pred) * scorer._sign
#     breakpoint()

if __name__ == '__main__':
    housing = fetch_california_housing()
    X, y = housing.data[:100], housing.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    estimator_id = 'random_forest'
    metric = 'mse'
    scorer = get_metric(metric)
    resampling_strategy = 'holdout'

    hpo = BaseFE(estimator_id=estimator_id,
                 metric=metric,
                 data_node=train_data, evaluation=resampling_strategy, resampling_params=None,
                 optimizer='tpe', per_run_time_limit=600,
                 time_limit=1024, amount_of_resource=5,
                 output_dir='./data', seed=1, n_jobs=1,
                 ensemble_method="blending", ensemble_size=5)

    print(hpo.run())
    breakpoint()
    y_true = test_data.data[1]
    pred_ens = hpo.predict(test_data, ens=True)
    pred = hpo.predict(test_data, ens=False)

    ens_perf = scorer._score_func(y_true, pred_ens) * scorer._sign
    pred_perf = scorer._score_func(y_true, pred) * scorer._sign
    breakpoint()
