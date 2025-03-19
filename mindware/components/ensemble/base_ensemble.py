from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import pickle as pkl
import time
import datetime

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.ensemble.unnamed_ensemble import choose_base_models_classification, \
    choose_base_models_regression
from mindware.components.feature_engineering.parse import construct_node
from mindware.utils.logging_utils import get_logger
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats, data_node,
                 ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 resampling_params=None,
                 output_dir=None, seed=None, ratio=0.5):
        self.stats = stats
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.resampling_params = resampling_params
        self.output_dir = output_dir
        self.seed = seed
        self.node = data_node
        self.ratio = ratio
        self.predictions = []
        self.train_labels = None
        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)
        
        self.if_imbal = False
        if self.task_type in CLS_TASKS:
            self.if_imbal = is_imbalanced_dataset(self.node)

        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        model_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                op_list, model, _ = CombinedTopKModelSaver._load(path)
                _node = self.node.copy_()
                _node = construct_node(_node, op_list)
                X, y = _node.data

                if self.task_type in CLS_TASKS:
                    ss = BaseCLSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=self.seed)
                else:
                    ss = BaseRGSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=self.seed)

                X_valid, y_valid = None, None
                for _, val_index in ss.split(X, y):
                    X_valid = X[val_index]
                    y_valid = y[val_index]

                if self.train_labels is not None:
                    assert np.all(self.train_labels == y_valid)
                else:
                    self.train_labels = y_valid

                if self.task_type in CLS_TASKS:
                    y_valid_pred = model.predict_proba(X_valid)
                else:
                    y_valid_pred = model.predict(X_valid)
                self.predictions.append(y_valid_pred)

                model_cnt += 1

        if len(self.predictions) < self.ensemble_size:
            self.ensemble_size = len(self.predictions)

        if ensemble_method == 'ensemble_selection':
            return

        if task_type in CLS_TASKS:
            self.base_model_mask = choose_base_models_classification(
                np.array(self.predictions), self.ensemble_size
            )
        else:
            self.base_model_mask = choose_base_models_regression(
                np.array(self.predictions), np.array(y_valid), self.ensemble_size, self.ratio
            )
        self.ensemble_size = sum(self.base_model_mask)

    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    # TODO: Refit
    def refit(self):
        raise NotImplementedError
