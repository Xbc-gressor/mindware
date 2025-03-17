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

    def __init__(self, stats, valid_data,
                 ensemble_method: str,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer,
                 output_dir=None, seed=None,
                 predictions=None):
        self.stats = stats
        self.valid_data = valid_data
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir
        self.seed = seed

        self.train_labels = None
        self.datetime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)
        
        self.if_imbal = if_imbal
        self.predictions = predictions
        self.base_model_mask = None

    def fit(self):
        raise NotImplementedError

    def predict(self, data, refit=False):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    # TODO: Refit
    def refit(self, datanode):
        raise NotImplementedError
