from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings
import os
import time
import datetime
import numpy as np
from sklearn.metrics._scorer import balanced_accuracy_scorer

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator

from mindware.components.utils.topk_saver import CombinedTopKModelSaver

from mindware.components.utils.constants import *
from mindware.components.feature_engineering.parse import construct_node


def get_kfold_name(folds, fold, seed, shuffle=False):

    if shuffle:
        return 'shu_s%d-k%d-no.%d' % (seed, folds, fold)
    else:
        return 'k%d-no.%d' % (folds, fold)


def parse_kfold_name(name):
    shuffle = False
    seed = None
    tmp = name.split('-')
    if 'shu_s' in name:
        shuffle = True
        seed = int(tmp[0][5:])
        tmp = tmp[1:]

    folds, fold = int(tmp[0][1:]), int(tmp[1][3:])

    return folds, fold, seed, shuffle

def fetch_predict_estimator(task_type, estimator_id, config, X_train, y_train, weight_balance=0, data_balance=0):
    # Build the ML estimator.
    from mindware.components.utils.balancing import get_weights, smote
    _fit_params = {}
    config_dict = config.copy()
    if task_type in CLS_TASKS and weight_balance == 1:
        _init_params, fit_params = get_weights(y_train, estimator_id, None, {}, {})
        for key, val in _init_params.items():
            config_dict[key] = val
        if 'sample_weight' in fit_params:
            _fit_params['sample_weight'] = fit_params['sample_weight']
        elif data_balance == 1:
            X_train, y_train = smote(X_train, y_train)
    if task_type in CLS_TASKS:
        from mindware.components.evaluators.cls_evaluator import get_estimator
    elif task_type in RGS_TASKS:
        from mindware.components.evaluators.rgs_evaluator import get_estimator
    _, estimator = get_estimator(config_dict, estimator_id)

    estimator.fit(X_train, y_train, **_fit_params)
    return estimator


def fetch_predict_results(task_type, op_list, estimator, test_node):
    if isinstance(estimator, dict):
        pred = None
        for key in estimator.keys():
            _op_list = op_list[key]
            _estimator = estimator[key]
            _test_node = test_node.copy_()
            _test_node = construct_node(_test_node, _op_list)
            if task_type in CLS_TASKS:
                _pred = _estimator.predict_proba(_test_node.data[0])
            else:
                _pred = _estimator.predict(_test_node.data[0])
            if pred is None:
                pred = _pred
            else:
                pred += _pred
        pred /= len(estimator)
    else:
        _test_node = test_node.copy_()
        _test_node = construct_node(_test_node, op_list)
        if task_type in CLS_TASKS:
            pred = estimator.predict_proba(_test_node.data[0])
        else:
            pred = estimator.predict(_test_node.data[0])

    return pred


class BaseEvaluator(_BaseEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, es=None
    ):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params
        self.es = es

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

        self.datetime = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

    @staticmethod
    def _get_train_valid_data(task_type, data_node, resampling_params=None, seed=1):

        test_size = 0.33
        if resampling_params is not None and 'test_size' in resampling_params:
            test_size = resampling_params['test_size']

        train_data = data_node.copy_(no_data=True)
        valid_data = data_node.copy_(no_data=True)
        X, y = data_node.data
        if task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=seed)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=seed)

        for train_index, val_index in ss.split(X, y):
            train_data.data = [X[train_index], y[train_index]]
            valid_data.data = [X[val_index], y[val_index]]

        return train_data, valid_data

    @staticmethod
    def _get_cv_data(task_type, data_node, resampling_params=None, seed=1, only_index=False):

        folds = 3
        if resampling_params is not None and 'folds' in resampling_params:
            folds = resampling_params['folds']

        X, y = data_node.data
        if task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy='cv', n_splits=folds, random_state=seed, shuffle=False)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy='cv', n_splits=folds, random_state=seed, shuffle=False)

        for train_index, val_index in ss.split(X, y):
            if only_index:
                yield(train_index, val_index)
            else:
                train_data = data_node.copy_(no_data=True)
                valid_data = data_node.copy_(no_data=True)
                train_data.data = [X[train_index], y[train_index]]
                valid_data.data = [X[val_index], y[val_index]]

                yield(train_data, valid_data, train_index, val_index)

    @staticmethod
    def _get_spliter(resampling_strategy, **kwargs):

        raise NotImplementedError
    
    def loss(self, es, x_val, y_val):
        pred_lst = []
        for model in es.model_pool:
            if not model:
                continue
            preds = model.predict(x_val)
            pred_lst.append(preds)
        pred_array = np.array(pred_lst)
        # model_num,sample
        if y_val.ndim == 1:
            y_val = y_val.reshape(1, -1)
        margin_loss = np.mean(pred_array * y_val, axis=-1)
        loss = 0
        for m_loss in margin_loss:
            loss += (1 - m_loss) ** 2 / 2
        return loss / len(margin_loss)

    def rgs_loss(self, es, x_val, y_val):
        pred_lst = []
        for model in es.model_pool:
            if not model:
                continue
            preds = model.predict(x_val)
            # 计算mse
            mse_loss = np.mean((preds - y_val) ** 2)
            pred_lst.append(mse_loss)
        # 所有模型mse的平均值
        return np.array(pred_lst).mean(axis=0)
    @staticmethod
    def _get_parse_data_node(config, train_node, val_node, if_imbal, record=True):

        return {}, train_node, val_node
    
    def train_estimator(self, config):
        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()
        if self.fixed_config is not None:
            config.update(self.fixed_config)
        estimator_id = config['algorithm']
        _x_train, _y_train = None, None
        # 默认用holdout
        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                train_node, val_node = self._get_train_valid_data(self.task_type, self.data_node, self.resampling_params, seed=self.seed)
                op_list, train_node, val_node = self._get_parse_data_node(config, train_node, val_node, self.if_imbal, record=True)

            _x_train, _y_train = train_node.data
            _x_val, _y_val = val_node.data

            estimator = fetch_predict_estimator(self.task_type, estimator_id=estimator_id, config=config,
                                                X_train=_x_train, y_train=_y_train,
                                                weight_balance=train_node.enable_balance, data_balance=train_node.data_balance)
        return estimator
    
    def __call__(self, config, es=None, **kwargs):

        start_time = time.time()
        return_dict = dict()
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
        # 默认用holdout
        if 'holdout' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                train_node, val_node = self._get_train_valid_data(self.task_type, self.data_node, self.resampling_params, seed=self.seed)
                op_list, train_node, val_node = self._get_parse_data_node(config, train_node, val_node, self.if_imbal, record=True)

            _x_train, _y_train = train_node.data
            _x_val, _y_val = val_node.data

            estimator = fetch_predict_estimator(self.task_type, estimator_id=estimator_id, config=config,
                                                X_train=_x_train, y_train=_y_train,
                                                weight_balance=train_node.enable_balance, data_balance=train_node.data_balance)
            
            es.replace_model(estimator)
            # 本身就是min问题，-一下后续再-一下就回来了，所以还是min问题
            if self.task_type in CLS_TASKS:
                score = -self.loss(es, _x_val, _y_val)
            else:
                score = -self.rgs_loss(es, _x_val, _y_val)

        elif 'cv' in self.resampling_strategy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                folds = 3
                if self.resampling_params is not None and 'folds' in self.resampling_params:
                    folds = self.resampling_params['folds']

                op_list_dict = dict()
                estimator_dict = dict()
                score_dict = dict()
                fold = 1
                for train_node, val_node, _, _ in self._get_cv_data(self.task_type, self.data_node, self.resampling_params, seed=self.seed):

                    op_list, train_node, val_node = self._get_parse_data_node(config, train_node, val_node, self.if_imbal, record=True)

                    _x_train, _y_train = train_node.data
                    _x_val, _y_val = val_node.data

                    estimator = fetch_predict_estimator(self.task_type, estimator_id=estimator_id, config=config, 
                                                        X_train=_x_train, y_train=_y_train, 
                                                        weight_balance=train_node.enable_balance, data_balance=train_node.data_balance)
                    score = self.scorer(estimator, _x_val, _y_val)

                    key = get_kfold_name(folds=folds, fold=fold, seed=self.seed, shuffle=False)
                    op_list_dict[key] = op_list
                    estimator_dict[key] = estimator
                    score_dict[key] = score
                    fold += 1
                score = np.mean(list(score_dict.values()))

                model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime, mode='cv', folds=folds)
                CombinedTopKModelSaver.save_config([op_list_dict, estimator_dict, score_dict], model_path)

                self.logger.info("Model saved to %s" % model_path)

        elif 'partial' in self.resampling_strategy:
            # Prepare data node.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                train_node, val_node = self._get_train_valid_data(self.task_type, self.data_node, self.resampling_params, seed=self.seed)
                op_list, train_node, val_node = self._get_parse_data_node(config, train_node, val_node, self.if_imbal, record=True)

            _x_train, _y_train = train_node.data

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

            _x_val, _y_val = val_node.data

            estimator = fetch_predict_estimator(self.task_type, estimator_id=estimator_id, config=config, 
                                                X_train=_act_x_train, y_train=_act_y_train, 
                                                weight_balance=train_node.enable_balance, data_balance=train_node.data_balance)
            score = self.scorer(estimator, _x_val, _y_val)

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
            if_imbal=False, es=None
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal, es
        )

    @staticmethod
    def _get_spliter(resampling_strategy, **kwargs):

        if 'cv' in resampling_strategy:
            folds = kwargs.pop('n_splits')
            shuffle = kwargs.pop('shuffle')
            random_state = kwargs.pop('random_state') if shuffle else None
            from sklearn.model_selection import StratifiedKFold
            return StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        elif 'holdout' in resampling_strategy or 'partial' in resampling_strategy:
            test_size = kwargs.pop('test_size')
            random_state = kwargs.pop('random_state')
            from sklearn.model_selection import StratifiedShuffleSplit
            return StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            raise ValueError('Invalid resampling strategy: %s!' % resampling_strategy)


class BaseRGSEvaluator(BaseEvaluator):

    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1, es=None
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal=False, es=es
        )

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


