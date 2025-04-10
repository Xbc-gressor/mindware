from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer, _PredictScorer
from sklearn.preprocessing import OneHotEncoder
from mindware.components.evaluators.base_evaluator import _BaseEvaluator

from mindware.utils.logging_utils import get_logger
import datetime
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
import os
import numpy as np
import warnings
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.ens.ens_utils import better_ens

class EnsEvaluator(_BaseEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, stats=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, val_nodes:dict=None,
    ):

        self.fixed_config = fixed_config
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.if_imbal = if_imbal
        self.task_type = task_type
        self.stats = stats
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = OneHotEncoder()
        if len(self.data_node.data[1].shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(self.data_node.data[1], (len(self.data_node.data[1]), 1))
            self.onehot_encoder.fit(reshape_y)
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.datetime = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.train_data = self.data_node.copy_(no_data=True)
        self.val_data = self.data_node.copy_(no_data=True)

        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        ss = self._get_spliter('holdout', test_size=test_size, random_state=self.seed)

        _x_train, _y_train = None, None
        _x_val, _y_val = None, None
        for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
            _x_train, _y_train = self.data_node.data[0][train_index], self.data_node.data[1][train_index]
            _x_val, _y_val = self.data_node.data[0][test_index], self.data_node.data[1][test_index]
        self.train_data.data = [_x_train, _y_train]
        self.val_data.data = [_x_val, _y_val]

        if val_nodes is None:
            val_nodes = {}
        val_nodes['val'] = self.val_data.copy_()
        self.val_nodes = val_nodes

        self.leader_board = dict()
        self.cache = dict()

        self.ensemble_builder = EnsembleBuilder(self.stats, self.val_data, self.task_type, self.scorer, resampling_params=self.resampling_params,
                                               output_dir=self.output_dir, seed=self.seed, if_imbal=self.if_imbal)

        self.best_model_path = None
        self.best_perf = -np.inf
        self.best_config = None
        self.comb_count = 0


    def _get_spliter(self, resampling_strategy, **kwargs):

        if self.task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)

        return ss

    def calculate_score(self, pred, y_true):
        if isinstance(self.scorer, _ThresholdScorer):
            if len(y_true.shape) == 1:
                y_true = self.onehot_encoder.transform(np.reshape(y_true, (len(y_true), 1))).toarray()
        elif self.task_type in CLS_TASKS and isinstance(self.scorer, _PredictScorer):
            pred = np.argmax(pred, axis=-1)
        score = self.scorer._score_func(y_true, pred) * self.scorer._sign
        return score

    def check_cache(self, base_model_mask):
        base_idx = np.where(base_model_mask)[0]
        key = '_'.join([str(x) for x in base_idx])

        if key in self.cache:
            return True, key
        else:
            return False, key

    def __call__(self, config, **kwargs):

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()

        if self.fixed_config is not None:
            config.update(self.fixed_config)

        # Prepare data node.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            base_model_mask = self.ensemble_builder.build_ensemble(ensemble_method='stacking', ensemble_size=config['ensemble_size'], ratio=config['ratio']/100,
                                                                stack_layers=config['stack_layers'], meta_learner=config['meta_learner'], opt=True)
            cache, key = self.check_cache(base_model_mask)
            if cache:
                learder_board = self.cache[key]
            else:
                stacking = self.ensemble_builder.fit(self.train_data, val_nodes=self.val_nodes)
                learder_board = stacking.leader_board
                self.cache[key] = learder_board
                self.comb_count += 1

                # best_perf = np.max(list(learder_board['val'].values()))
                best_config = self.ensemble_builder.model.best_config.copy()
                best_config.update({'ensemble_size': config['ensemble_size'], 'ratio': config['ratio']})
                if better_ens(best_config, self.best_config):
                    self.best_config = best_config.copy()
                    model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime)
                    if self.best_model_path is not None:
                        os.remove(self.best_model_path)
                        self.logger.info(f"Remove old best ens model: {self.best_model_path}!")
                    self.logger.info(f"Save new best ens model: {model_path}!")
                    CombinedTopKModelSaver.save_config([None, self.ensemble_builder, learder_board], model_path)
                    self.best_perf = best_config['val']
                    self.best_model_path = model_path

        size = config['ensemble_size']
        ratio = config['ratio']
        base = f'ens{size}_r{ratio}'
        for key in learder_board:
            if key not in self.leader_board:
                self.leader_board[key] = {}
            for leader, objective in learder_board[key].items():
                head = f"{base}-{leader}"
                assert head not in self.leader_board[key]
                self.leader_board[key][head] = objective

        # 取反
        learder_board = {key: -value for key, value in learder_board['val'].items()}
        return_dict = dict()
        # Turn it into a minimization problem.
        return_dict['leader_board'] = learder_board

        return return_dict