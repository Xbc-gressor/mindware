from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings
import os
import time
import numpy as np
import pickle as pkl
import datetime
from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.constants import *
from mindware.components.evaluators.evaluate_func import validation

from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator
from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator

from mindware.modules.base_evaluator import BaseCLSEvaluator
from mindware.modules.base_evaluator import BaseRGSEvaluator


class FECLSEvaluator(BaseCLSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal
        )

    def _get_parse_data_node(self, config, record=True):
        data_node, op_list = parse_config(self.train_node, config, record=record, if_imbal=self.if_imbal)
        _val_node = self.val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node


class FERGSEvaluator(BaseRGSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed
        )

    def _get_parse_data_node(self, config, record=True):
        data_node, op_list = parse_config(self.train_node, config, record=record)
        _val_node = self.val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node


class FEClassificationEvaluator(_BaseEvaluator):

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

        self.seed = 1
        self.timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()

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

        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']

                from sklearn.model_selection import StratifiedShuffleSplit
                ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                data_node, op_list = parse_config(self.train_node, config, record=True, if_imbal=self.if_imbal)
                _val_node = self.val_node.copy_()
                _val_node = construct_node(_val_node, op_list)

            _x_train, _y_train = data_node.data
            _x_val, _y_val = _val_node.data

            y = np.reshape(_y_train, (len(_y_train), 1))
            self.onehot_encoder.fit(y)

            score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                               random_state=self.seed,
                               onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                        _ThresholdScorer) else None)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                from sklearn.model_selection import StratifiedKFold
                skfold = StratifiedKFold(n_splits=folds, random_state=self.seed, shuffle=False)
                scores = list()

                for train_index, test_index in skfold.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True, if_imbal=self.if_imbal)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                    _x_train, _y_train = data_node.data
                    _x_val, _y_val = _val_node.data

                    y = np.reshape(_y_train, (len(_y_train), 1))
                    self.onehot_encoder.fit(y)

                    _score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                        random_state=self.seed,
                                        onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                 _ThresholdScorer) else None)
                    scores.append(_score)
                score = np.mean(scores)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']

                from sklearn.model_selection import StratifiedShuffleSplit
                ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                data_node, op_list = parse_config(self.train_node, config, record=True, if_imbal=self.if_imbal)
                _val_node = self.val_node.copy_()
                _val_node = construct_node(_val_node, op_list)

            _x_train, _y_train = data_node.data

            if downsample_ratio != 1:
                down_ss = StratifiedShuffleSplit(n_splits=1, test_size=downsample_ratio,
                                                 random_state=self.seed)
                for _, _val_index in down_ss.split(_x_train, _y_train):
                    _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
            else:
                _act_x_train, _act_y_train = _x_train, _y_train
                _val_index = list(range(len(_x_train)))

            _x_val, _y_val = _val_node.data

            y = np.reshape(_y_train, (len(_y_train), 1))
            self.onehot_encoder.fit(y)

            score = validation(clf, self.scorer, _act_x_train, _act_y_train, _x_val, _y_val,
                               random_state=self.seed,
                               onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                        _ThresholdScorer) else None)


        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:
            if np.isfinite(score) and downsample_ratio == 1:
                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)

                if not os.path.exists(model_path):
                    with open(model_path, 'wb') as f:
                        pkl.dump([op_list, clf, score], f)
                else:
                    with open(model_path, 'rb') as f:
                        _, _, perf = pkl.load(f)
                    if score > perf:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, clf, score], f)

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


class FERegressionEvaluator(_BaseEvaluator):

    def __init__(self, estimator_id=None, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION,
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

        self.timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()

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

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']

                from sklearn.model_selection import ShuffleSplit
                ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                data_node, op_list = parse_config(self.train_node, config, record=True)
                _val_node = self.val_node.copy_()
                _val_node = construct_node(_val_node, op_list)

            _x_train, _y_train = data_node.data
            _x_val, _y_val = _val_node.data

            score = validation(rgs, self.scorer, _x_train, _y_train, _x_val, _y_val,
                               random_state=self.seed)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                from sklearn.model_selection import KFold
                kfold = KFold(n_splits=folds, random_state=self.seed, shuffle=False)
                scores = list()

                for train_index, test_index in kfold.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                    _x_train, _y_train = data_node.data
                    _x_val, _y_val = _val_node.data

                    _score = validation(rgs, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                        random_state=self.seed)
                    scores.append(_score)
                score = np.mean(scores)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']

                from sklearn.model_selection import ShuffleSplit
                ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                data_node, op_list = parse_config(self.train_node, config, record=True)
                _val_node = self.val_node.copy_()
                _val_node = construct_node(_val_node, op_list)

            _x_train, _y_train = data_node.data

            if downsample_ratio != 1:
                down_ss = ShuffleSplit(n_splits=1, test_size=downsample_ratio,
                                       random_state=self.seed)
                for _, _val_index in down_ss.split(_x_train, _y_train):
                    _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
            else:
                _act_x_train, _act_y_train = _x_train, _y_train
                _val_index = list(range(len(_x_train)))

            _x_val, _y_val = _val_node.data

            score = validation(rgs, self.scorer, _act_x_train, _act_y_train, _x_val, _y_val,
                               random_state=self.seed)


        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:
            if np.isfinite(score) and downsample_ratio == 1:
                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)

                if not os.path.exists(model_path):
                    with open(model_path, 'wb') as f:
                        pkl.dump([op_list, rgs, score], f)
                else:
                    with open(model_path, 'rb') as f:
                        _, _, perf = pkl.load(f)
                    if score > perf:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, rgs, score], f)

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
