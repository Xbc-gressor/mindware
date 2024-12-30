from sklearn.metrics._scorer import _BaseScorer
from mindware.components.ensemble.bagging import Bagging
from mindware.components.ensemble.blending import Blending
from mindware.components.ensemble.crossvalidation_ensemble import CrossValidationEnsembleModel
from mindware.components.ensemble.stacking import Stacking
from mindware.components.ensemble.ensemble_selection import EnsembleSelection

ensemble_list = ['bagging', 'blending', 'stacking', 'ensemble_selection', 'cross_validation']


class EnsembleBuilder:
    def __init__(self, stats, data_node,
                 ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 resampling_params=None,
                 output_dir=None, seed=None):
        self.model = None
        if ensemble_method == 'bagging':
            self.model = Bagging(stats=stats,
                                 data_node=data_node,
                                 ensemble_size=ensemble_size,
                                 task_type=task_type,
                                 metric=metric,
                                 resampling_params=resampling_params,
                                 output_dir=output_dir, seed=seed)
        elif ensemble_method == 'blending':
            self.model = Blending(stats=stats,
                                  data_node=data_node,
                                  ensemble_size=ensemble_size,
                                  task_type=task_type,
                                  metric=metric,
                                  resampling_params=resampling_params,
                                  output_dir=output_dir, seed=seed)
        elif ensemble_method == 'stacking':
            self.model = Stacking(stats=stats,
                                  data_node=data_node,
                                  ensemble_size=ensemble_size,
                                  task_type=task_type,
                                  metric=metric,
                                  resampling_params=resampling_params,
                                  output_dir=output_dir, seed=seed)
        elif ensemble_method == 'ensemble_selection':
            self.model = EnsembleSelection(stats=stats,
                                           data_node=data_node,
                                           ensemble_size=ensemble_size,
                                           task_type=task_type,
                                           metric=metric,
                                           resampling_params=resampling_params,
                                           output_dir=output_dir, seed=seed)
        elif ensemble_method == 'cross_validation':
            self.model = CrossValidationEnsembleModel(stats=stats,
                                                      data_node=data_node,
                                                      ensemble_size=ensemble_size,
                                                      task_type=task_type,
                                                      metric=metric,
                                                      resampling_params=resampling_params,
                                                      output_dir=output_dir, seed=seed)
        else:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def fit(self, data):
        return self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def refit(self):
        return self.model.refit()

    def get_ens_model_info(self):
        return self.model.get_ens_model_info()
