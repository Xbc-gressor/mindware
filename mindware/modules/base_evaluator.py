from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
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
from mindware.components.evaluators.evaluate_func import validation

from mindware.components.utils.topk_saver import CombinedTopKModelSaver

from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator
from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator


class BaseEvaluator(_BaseEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False
    ):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

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

        self.train_node = None
        self.val_node = None

        self.datetime = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

    def get_fit_params(self, y, estimator):
        from mindware.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def _get_estimator_getter(self):

        raise NotImplementedError

    def _get_spliter(self, resampling_strategy, **kwargs):

        raise NotImplementedError

    def _get_parse_data_node(self, config, record=True):

        raise NotImplementedError

    def _get_onehot_encoder(self, y):

        raise NotImplementedError

    def __call__(self, config, **kwargs):
        
        self.train_node = self.data_node.copy_(no_data=True)
        self.val_node = self.data_node.copy_(no_data=True)

        start_time = time.time()
        return_dict = dict()
        self.seed = 1
        downsample_ratio = kwargs.get('resource_ratio', 1.0)

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()
        if self.fixed_config is not None:
            config.update(self.fixed_config)
        estimator_id = config['algorithm']

        score = -np.inf
        estimator = None
        _x_train, _y_train = None, None
        _act_x_train, _act_y_train = None, None
        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']
                    
                ss = self.__class__._get_spliter(self.resampling_strategy, test_size=test_size, random_state=self.seed)

                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                op_list, data_node, _val_node = self._get_parse_data_node(config, record=True)

            _x_train, _y_train = data_node.data
            _x_val, _y_val = _val_node.data

            config_dict = config.copy()
            # Prepare training and initial params for classifier.
            init_params, fit_params = {}, {}
            if data_node.enable_balance == 1:
                init_params, fit_params = self.get_fit_params(_y_train, estimator_id)
            if init_params is not None:
                for key, val in init_params.items():
                    config_dict[key] = val
                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

            _, estimator = self._get_estimator_getter()(config_dict, estimator_id)

            # if self.onehot_encoder is None:
            #     self.onehot_encoder = self._get_onehot_encoder(_y_train)

            score = validation(estimator, self.scorer, _x_train, _y_train, _x_val, _y_val,
                               random_state=self.seed,
                               onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                        _ThresholdScorer) else None,
                               fit_params=fit_params)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if 'cv' in self.resampling_strategy:
                    if self.resampling_params is None or 'folds' not in self.resampling_params:
                        folds = 5
                    else:
                        folds = self.resampling_params['folds']

                skfold = self.__class__._get_spliter(self.resampling_strategy,
                                           n_splits=folds, random_state=self.seed, shuffle=False)
                scores = list()
                for train_index, test_index in skfold.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    _, data_node, _val_node = self._get_parse_data_node(config, record=True)

                    _x_train, _y_train = data_node.data
                    _x_val, _y_val = _val_node.data

                    config_dict = config.copy()
                    # Prepare training and initial params for classifier.
                    init_params, fit_params = {}, {}
                    if data_node.enable_balance == 1:
                        init_params, fit_params = self.get_fit_params(_y_train, estimator_id)
                    if init_params is not None:
                        for key, val in init_params.items():
                            config_dict[key] = val
                        if data_node.data_balance == 1:
                            fit_params['data_balance'] = True

                    _, estimator = self._get_estimator_getter()(config_dict, estimator_id)

                    # if self.onehot_encoder is None:
                    #     self.onehot_encoder = self._get_onehot_encoder(_y_train)

                    _score = validation(estimator, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                        random_state=self.seed,
                                        onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                 _ThresholdScorer) else None,
                                        fit_params=fit_params)
                    scores.append(_score)
                score = np.mean(scores)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                test_size = 0.33
                if self.resampling_params is not None and 'test_size' in self.resampling_params:
                    test_size = self.resampling_params['test_size']

                ss = self.__class__._get_spliter(self.resampling_strategy, test_size=test_size, random_state=self.seed)
                for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                    _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                    _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                self.train_node.data = [_x_train, _y_train]
                self.val_node.data = [_x_val, _y_val]

                op_list, data_node, _val_node = self._get_parse_data_node(config, record=True)

            _x_train, _y_train = data_node.data

            _val_index = None
            _act_x_train, _act_y_train = None, None
            if downsample_ratio != 1:
                down_ss = self.__class__._get_spliter(self.resampling_strategy, test_size=downsample_ratio,
                                            random_state=self.seed)
                for _, _val_index in down_ss.split(_x_train, _y_train):
                    _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
            else:
                _act_x_train, _act_y_train = _x_train, _y_train
                _val_index = list(range(len(_x_train)))

            _x_val, _y_val = _val_node.data

            config_dict = config.copy()
            # Prepare training and initial params for classifier.
            init_params, fit_params = {}, {}
            if data_node.enable_balance == 1:
                init_params, fit_params = self.get_fit_params(_y_train, estimator_id)
            if init_params is not None:
                for key, val in init_params.items():
                    config_dict[key] = val
                if 'sample_weight' in fit_params:
                    fit_params['sample_weight'] = fit_params['sample_weight'][_val_index]
                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

            _, estimator = self._get_estimator_getter()(config_dict, estimator_id)

            if self.onehot_encoder is None:
                self.onehot_encoder = self._get_onehot_encoder(_y_train)

            score = validation(estimator, self.scorer, _act_x_train, _act_y_train, _x_val, _y_val,
                               random_state=self.seed,
                               onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                        _ThresholdScorer) else None,
                               fit_params=fit_params)

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        if 'holdout' in self.resampling_strategy or 'partial' in self.resampling_strategy:

            if np.isfinite(score) and downsample_ratio == 1:
                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime)
                CombinedTopKModelSaver.save_config([op_list, estimator, score], model_path)

                self.logger.info("Model saved to %s" % model_path)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (estimator_id,
                              self.scorer._sign * score,
                              time.time() - start_time,
                              _x_train.shape if 'partial' not in self.resampling_strategy else _act_x_train.shape))
        except:
            pass

        # Turn it into a minimization problem.
        return_dict['objectives'] = [-score]

        return return_dict


class BaseCLSEvaluator(BaseEvaluator):

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

    def get_fit_params(self, y, estimator):
        from mindware.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def _get_estimator_getter(self):
        return get_cls_estimator

    @staticmethod
    def _get_spliter(resampling_strategy, **kwargs):

        if 'cv' in resampling_strategy:
            folds = kwargs.pop('n_splits')
            shuffle = kwargs.pop('shuffle')
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=folds, shuffle=shuffle)
        elif 'holdout' in resampling_strategy or 'partial' in resampling_strategy:
            test_size = kwargs.pop('test_size')
            random_state = kwargs.pop('random_state')
            from sklearn.model_selection import StratifiedShuffleSplit
            return StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            raise ValueError('Invalid resampling strategy: %s!' % resampling_strategy)

    def _get_parse_data_node(self, config, record=True):
        return {}, self.train_node, self.val_node

    def _get_onehot_encoder(self, y):
        onehot_encoder = OneHotEncoder(categories='auto')
        y = np.reshape(y, (len(y), 1))
        onehot_encoder.fit(y)
        return onehot_encoder


class BaseRGSEvaluator(BaseEvaluator):

    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal=False
        )

    def get_fit_params(self, y, estimator):

        return None, None

    def _get_estimator_getter(self):
        return get_rgs_estimator

    @staticmethod
    def _get_spliter(resampling_strategy, **kwargs):

        if 'cv' in resampling_strategy:
            folds = kwargs.pop('n_splits')
            shuffle = kwargs.pop('shuffle')
            from sklearn.model_selection import KFold
            return KFold(n_splits=folds, shuffle=shuffle)
        elif 'holdout' in resampling_strategy or 'partial' in resampling_strategy:
            test_size = kwargs.pop('test_size')
            random_state = kwargs.pop('random_state')
            from sklearn.model_selection import ShuffleSplit
            return ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            raise ValueError('Invalid resampling strategy: %s!' % resampling_strategy)

    def _get_parse_data_node(self, config, record=True):
        return {}, self.train_node, self.val_node

    def _get_onehot_encoder(self, y):

        return None
