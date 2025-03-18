from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer, _PredictScorer
from sklearn.preprocessing import OneHotEncoder
from mindware.components.evaluators.base_evaluator import _BaseEvaluator

from mindware.utils.logging_utils import get_logger
import datetime
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
import time
import numpy as np
import warnings

class EnsEvaluator(_BaseEvaluator):
    def __init__(
            self, scorer=None, stats=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False
    ):
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
            reshape_y = np.reshape(self.data_node.data[1].shape, (len(self.data_node.data[1].shape), 1))
            self.onehot_encoder.fit(reshape_y)
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.datetime = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.val_data = self.data_node.copy_(no_data=True)

        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        ss = self._get_spliter(self.resampling_strategy, test_size=test_size, random_state=self.seed)

        _x_val, _y_val = None, None
        for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
            _x_val, _y_val = self.data_node.data[0][test_index], self.data_node.data[1][test_index]
        self.val_data.data = [_x_val, _y_val]

        self.n_fold = 5
        self.split_index = []
        skfold = self._get_spliter('cv', n_splits=self.n_fold, shuffle=False, random_state=self.seed)
        self.ensemble_builders = []
        predictions = np.array(EnsembleBuilder.build_predictions(self.stats, self.val_data, self.task_type))
        for train_index, test_index in skfold.split(self.val_data.data[0], self.val_data.data[1]):
            val_train_data = self.data_node.copy_(no_data=True)
            val_train_data.data = [self.val_data.data[0][train_index], self.val_data.data[1][train_index]]
            ensemble_builder = EnsembleBuilder(self.stats, val_train_data, self.task_type, self.scorer,
                                               output_dir=self.output_dir, seed=self.seed, if_imbal=self.if_imbal, _predict=False)
            ensemble_builder.predictions = predictions[:, train_index]
            self.split_index.append((train_index, test_index))
            self.ensemble_builders.append(ensemble_builder)


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

    def __call__(self, config, **kwargs):

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()

        # Prepare data node.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            final_pred = None
            for i, (_, test_index) in enumerate(self.split_index):
                val_val_node = self.val_data.copy_(no_data=True)
                val_val_node.data = [self.val_data.data[0][test_index], self.val_data.data[1][test_index]]
                self.ensemble_builders[i].build_ensemble(**config)
                self.ensemble_builders[i].fit()
                pred = self.ensemble_builders[i].predict(val_val_node)
                if final_pred is None:
                    if len(pred.shape) == 1:
                        final_pred = np.zeros(len(self.val_data.data[1]))
                    else:
                        final_pred = np.zeros((len(self.val_data.data[1]), pred.shape[1]))

                final_pred[test_index] = pred

        score = self.calculate_score(final_pred, self.val_data.data[1])
        return_dict = dict()
        # Turn it into a minimization problem.
        return_dict['objectives'] = [-score]