from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings
import os
import time
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation
from mindware.components.feature_engineering.task_space import get_task_hyperparameter_space
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons
from mindware.components.utils.constants import *

from mindware.components.metrics.metric import get_metric

from mindware.components.evaluators.cls_evaluator import get_estimator


class HPOClassificationEvaluator(_BaseEvaluator):

    def __init__(self, estimator_id=None, fixed_config=None, scorer=None, data_node=None, task_type=0,
                 resampling_strategy='cv', resampling_params=None,
                 timestamp=None, output_dir=None, seed=1, if_imbal=False):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.estimator_id = estimator_id
        self.fixed_config = fixed_config
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.if_imbal = if_imbal
        self.task_type = task_type
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.continue_training = False

        self.seed = 1
        self.timestamp = timestamp

    def __call__(self, config, **kwargs):
        start_time = time.time()
        result_dict = dict()
        downsample_ratio = kwargs.get('resource_ratio', 1.0)

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()
        if self.fixed_config is not None:
            config.update(self.fixed_config)

        # X, y Data
        _x, _y = self.data_node.data

        # Prepare training and initial params for classifier.
        init_params, fit_params = {}, {}
        _candidates = get_combined_candidtates(_classifiers, _addons)
        clf = _candidates[self.estimator_id](**config)

        # One-hot encoder
        if self.onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(categories='auto')
            y = np.reshape(_y, (len(_y), 1))
            self.onehot_encoder.fit(y)

        onehot = self.onehot_encoder if isinstance(self.scorer, _ThresholdScorer) else None

        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']

            score = holdout_validation(clf, self.scorer,
                                       self.data_node.data[0], self.data_node.data[1], test_size=test_size,
                                       fit_params=fit_params, if_stratify=True, onehot=onehot, random_state=self.seed)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                score = cross_validation(clf, self.scorer,
                                         _x, _y, n_fold=folds,
                                         shuffle=False, fit_params=fit_params, if_stratify=True, onehot=onehot,
                                         random_state=self.seed)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']

            score = partial_validation(clf, self.scorer,
                                       _x, _y, data_subsample_ratio=downsample_ratio, test_size=test_size,
                                       fit_params=fit_params, if_stratify=True, onehot=onehot, random_state=self.seed)

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        # if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:
        #     if np.isfinite(score) and downsample_ratio == 1:
        #         model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)
        #
        #         if not os.path.exists(model_path):
        #             with open(model_path, 'wb') as f:
        #                 pkl.dump([None, clf, score], f)
        #         else:
        #             with open(model_path, 'rb') as f:
        #                 _, _, perf = pkl.load(f)
        #             if score > perf:
        #                 with open(model_path, 'wb') as f:
        #                     pkl.dump([None, clf, score], f)
        #
        #         self.logger.info("Model saved to %s" % model_path)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (self.estimator_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _x.shape))
        except:
            pass

        # Turn it into a minimization problem.
        result_dict['objectives'] = [-score]

        return result_dict
