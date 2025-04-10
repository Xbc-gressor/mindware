from sklearn.metrics._scorer import _BaseScorer
from mindware.components.ensemble.bagging import Bagging
from mindware.components.ensemble.blending import Blending
from mindware.components.ensemble.crossvalidation_ensemble import CrossValidationEnsembleModel
from mindware.components.ensemble.stacking import Stacking
from mindware.components.ensemble.ensemble_selection import EnsembleSelection
from mindware.components.utils.topk_saver import CombinedTopKModelSaver, check_mode
from mindware.components.feature_engineering.parse import construct_node
from mindware.components.utils.constants import CLS_TASKS
import numpy as np
from mindware.utils.logging_utils import get_logger

from mindware.components.ensemble.unnamed_ensemble import choose_base_models_classification, \
    choose_base_models_regression

ensemble_list = ['bagging', 'blending', 'stacking', 'ensemble_selection', 'cross_validation']


class EnsembleBuilder:
    def __init__(self, stats, valid_node,
                 task_type: int, metric: _BaseScorer,
                 resampling_params = None,
                 output_dir=None, seed=None,
                 if_imbal=False,
                 ):
        self.model = None
        self.stats = stats
        self.valid_node = valid_node.copy_()
        self.task_type = task_type
        self.metric = metric
        self.resampling_params = resampling_params
        self.output_dir = output_dir
        self.seed = seed

        self.predictions = []
        self.model = None
        self.if_imbal = if_imbal

        self.predictions = self.build_predictions(stats, valid_node, task_type)

        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)

    def build_ensemble(self, ensemble_method, ensemble_size, **kwargs):
        ratio = kwargs.get('ratio', 0.4)
        self.logger.info(f"Ensemble size: {ensemble_size}, ratio: {ratio}")
        stack_layers = kwargs.get('stack_layers', 5)
        meta_learner = kwargs.get('meta_learner', 'auto')

        if len(self.predictions) < ensemble_size:
            self.logger.info("The number of models is less than the ensemble size. "
                             "Set the ensemble size to the number of models.")
            ensemble_size = len(self.predictions)

        base_model_mask = None
        if ensemble_method not in ["ensemble_selection"]:
            y_valid = self.valid_node.data[1]
            if self.task_type in CLS_TASKS:
                base_model_mask = choose_base_models_classification(
                    np.array(self.predictions), np.array(y_valid), ensemble_size, ratio=ratio
                )
            else:
                base_model_mask = choose_base_models_regression(
                    np.array(self.predictions), np.array(y_valid), ensemble_size, ratio=ratio
                )
            ensemble_size = sum(base_model_mask)

        if ensemble_method == 'bagging':
            self.model = Bagging(stats=self.stats,
                                 ensemble_size=ensemble_size,
                                 task_type=self.task_type, if_imbal=self.if_imbal,
                                 metric=self.metric, resampling_params=self.resampling_params,
                                 output_dir=self.output_dir, seed=self.seed,
                                 predictions=self.predictions, base_model_mask=base_model_mask)
        elif ensemble_method == 'blending':
            self.model = Blending(stats=self.stats,
                                  ensemble_size=ensemble_size,
                                  task_type=self.task_type, if_imbal=self.if_imbal,
                                  metric=self.metric, resampling_params=self.resampling_params,
                                  output_dir=self.output_dir, seed=self.seed,
                                  stack_layers=stack_layers, meta_learner=meta_learner,
                                  predictions=self.predictions, base_model_mask=base_model_mask)
        elif ensemble_method == 'stacking':
            self.model = Stacking(stats=self.stats,
                                  ensemble_size=ensemble_size,
                                  task_type=self.task_type, if_imbal=self.if_imbal,
                                  metric=self.metric, resampling_params=self.resampling_params,
                                  output_dir=self.output_dir, seed=self.seed,
                                  stack_layers=stack_layers, meta_learner=meta_learner,
                                  predictions=self.predictions, base_model_mask=base_model_mask,
                                  opt=kwargs.get('opt', False))
        elif ensemble_method == 'ensemble_selection':
            self.model = EnsembleSelection(stats=self.stats,
                                           ensemble_size=ensemble_size,
                                           task_type=self.task_type, if_imbal=self.if_imbal,
                                           metric=self.metric, resampling_params=self.resampling_params,
                                           output_dir=self.output_dir, seed=self.seed,
                                           predictions=self.predictions)
        else:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)
        
        return base_model_mask

    @staticmethod
    def build_predictions(stats, valid_node, task_type):
        predictions = []
        model_cnt = 0
        for algo_id in stats.keys():
            model_to_eval = stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                op_list, model, _ = CombinedTopKModelSaver._load(path)
                _node = valid_node.copy_()
                _node = construct_node(_node, op_list)
                X_valid, y_valid = _node.data

                if task_type in CLS_TASKS:
                    y_valid_pred = model.predict_proba(X_valid)
                else:
                    y_valid_pred = model.predict(X_valid)
                predictions.append(y_valid_pred)

                model_cnt += 1

        return predictions

    def fit(self, datanode, **kwargs):
        return self.model.fit(datanode, **kwargs)

    def predict(self, data, refit):
        check_mode(refit)
        return self.model.predict(data, refit)

    def refit(self, datanode, mode):
        check_mode(mode)
        assert mode != 'partial'
        return self.model.refit(datanode, mode=mode)

    def get_ens_model_info(self):
        return self.model.get_ens_model_info()
