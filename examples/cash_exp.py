import argparse
import os
import sys
import time

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from mindware.utils.data_manager import DataManager
from mindware.modules.cash.base_cash import BaseCASHOptimizer

from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *




if __name__ == '__main__':

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    dm = DataManager(X_train, y_train)
    train_data = dm.get_data_node(X_train, y_train)
    test_data = dm.get_data_node(X_test, y_test)

    estimator_id = 'adaboost'
    scorer = get_metric('acc')
    resampling_strategy = 'holdout'

#     timestamp = time.time()
#     clf_class = _classifiers[estimator_id]
#     cs = clf_class.get_hyperparameter_search_space()
#     config = cs.sample_configuration().get_dictionary()
#     hpo_eva = HPOClassificationEvaluator(estimator_id=estimator_id,
#                                          fixed_config=None,
#                                          scorer=scorer,
#                                          data_node=train_data,
#                                          resampling_strategy=resampling_strategy,
#                                          timestamp=timestamp,
#                                          output_dir='./data',
#                                          seed=1,
#                                          if_imbal=False)
#
#     print(hpo_eva(config))
    hpo = BaseCASHOptimizer(
                           task_type=CLASSIFICATION,
                           scorer=scorer,
                           data_node=train_data, evaluation='holdout', resampling_params=None,
                           optimizer='smac', per_run_time_limit=600,
                           time_limit=1024, amount_of_resource=50,
                           output_dir='../data', seed=1, n_jobs=1)
#     # smac, random_search, tpe, partial_bohb
#     # /Users/xubeideng/Documents/Scientific Research/AutoML/code/mindware/data
#     # D:\\xbc\\Fighting\\AutoML\\codes\\mindware\\data
#
    print(hpo.run())


# if __name__ == '__main__':
#     housing = fetch_california_housing()
#
#     X, y = housing.data, housing.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#     dm = DataManager(X_train, y_train)
#     train_data = dm.get_data_node(X_train, y_train)
#     test_data = dm.get_data_node(X_test, y_test)
#
#     estimator_id = 'adaboost'
#     scorer = get_metric('mse')
#     resampling_strategy = 'holdout'
#
#     hpo = BaseCASHOptimizer(
#                            task_type=REGRESSION,
#                            scorer=scorer,
#                            data_node=train_data, evaluation='holdout', resampling_params=None,
#                            optimizer='tpe', per_run_time_limit=600,
#                            time_limit=1024, amount_of_resource=20,
#                            output_dir='../data', seed=1, n_jobs=1
#     )
#
#     print(hpo.run())

