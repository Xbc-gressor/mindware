import warnings
import os
import time
import datetime
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons as _cls_addons
from mindware.components.models.regression import _regressors, _addons as _rgs_addons
from mindware.components.utils.constants import *
from ConfigSpace import ConfigurationSpace, Constant


from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator
from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator


def get_hpo_cs(estimator_id, task_type):
    if task_type in CLS_TASKS:
        _candidates = get_combined_candidtates(_classifiers, _cls_addons)
    else:
        _candidates = get_combined_candidtates(_regressors, _rgs_addons)

    if estimator_id in _candidates:
        rgs_class = _candidates[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)

    tmp_cs = rgs_class.get_hyperparameter_search_space()
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Constant('algorithm', estimator_id))

    for hyper in tmp_cs.get_hyperparameters():
        hyper.name = '%s:%s' % (estimator_id, hyper.name)
        cs.add_hyperparameter(hyper)

    return cs


# def get_hpo_conf(config, estimator_id):
#     config_ = config.copy()
#     hpo_config = dict()
#     for key in config_:
#         key_name = key.split(':')[0]
#         if estimator_id == key_name:
#             act_key = key.split(':')[1]
#             hpo_config[act_key] = config_[key]
#
#     return hpo_config


class HPOClassificationEvaluator(_BaseEvaluator):

    def __init__(self, estimator_id=None, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION,
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

        self.seed = seed
        self.timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

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
        _X, _y = self.data_node.data

        assert self.estimator_id == config['algorithm']
        config_dict = config.copy()
        _, clf = get_cls_estimator(config_dict, self.estimator_id)

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
                                       _X, _y, test_size=test_size,
                                       if_stratify=True, onehot=onehot, random_state=self.seed)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                score = cross_validation(clf, self.scorer,
                                         _X, _y, n_fold=folds,
                                         shuffle=False,
                                         if_stratify=True, onehot=onehot, random_state=self.seed)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']

            score = partial_validation(clf, self.scorer,
                                       _X, _y, data_subsample_ratio=downsample_ratio, test_size=test_size,
                                       if_stratify=True, onehot=onehot, random_state=self.seed)

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:
            if np.isfinite(score) and downsample_ratio == 1:
                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)

                if not os.path.exists(model_path):
                    with open(model_path, 'wb') as f:
                        pkl.dump([{}, clf, score], f)
                else:
                    with open(model_path, 'rb') as f:
                        _, _, perf = pkl.load(f)
                    if score > perf:
                        with open(model_path, 'wb') as f:
                            pkl.dump([{}, clf, score], f)

                self.logger.info("Model saved to %s" % model_path)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (self.estimator_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _X.shape))
        except:
            pass

        # Turn it into a minimization problem.
        result_dict['objectives'] = [-score]

        return result_dict


class HPORegressionEvaluator(_BaseEvaluator):

    def __init__(self, estimator_id=None, fixed_config=None, scorer=None, data_node=None, task_type=REGRESSION,
                 resampling_strategy='cv', resampling_params=None,
                 timestamp=None, output_dir=None, seed=1):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.estimator_id = estimator_id
        self.fixed_config = fixed_config
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.task_type = task_type
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.continue_training = False

        self.seed = seed
        self.timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

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
        _X, _y = self.data_node.data

        assert self.estimator_id == config['algorithm']
        config_dict = config.copy()
        _, rgs = get_rgs_estimator(config_dict, self.estimator_id)

        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']

            score = holdout_validation(rgs, self.scorer,
                                       _X, _y, test_size=test_size,
                                       if_stratify=False, random_state=self.seed)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                score = cross_validation(rgs, self.scorer,
                                         _X, _y, n_fold=folds,
                                         shuffle=False,
                                         if_stratify=False, random_state=self.seed)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if self.resampling_params is None or 'test_size' not in self.resampling_params:
                    test_size = 0.33
                else:
                    test_size = self.resampling_params['test_size']

            score = partial_validation(rgs, self.scorer,
                                       _X, _y, data_subsample_ratio=downsample_ratio, test_size=test_size,
                                       if_stratify=False, random_state=self.seed)

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:
            if np.isfinite(score) and downsample_ratio == 1:
                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)

                if not os.path.exists(model_path):
                    with open(model_path, 'wb') as f:
                        pkl.dump([{}, rgs, score], f)
                else:
                    with open(model_path, 'rb') as f:
                        _, _, perf = pkl.load(f)
                    if score > perf:
                        with open(model_path, 'wb') as f:
                            pkl.dump([{}, rgs, score], f)

                self.logger.info("Model saved to %s" % model_path)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (self.estimator_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _X.shape))
        except:
            pass

        # Turn it into a minimization problem.
        result_dict['objectives'] = [-score]

        return result_dict
