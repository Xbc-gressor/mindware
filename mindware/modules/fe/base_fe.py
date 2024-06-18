import os

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.feature_engineering.task_space import get_task_hyperparameter_space
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons as _cls_addons
from mindware.components.models.regression import _regressors, _addons as _rgs_addons

from mindware.components.config_space.cs_builder import get_fe_cs

from ConfigSpace import Configuration, Constant


class BaseFE(BaseAutoML):
    def __init__(self, estimator_id: str, task_type: str = None,
                 metric: str = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac',
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method=None, ensemble_size=None,
                 include_preprocessors=None, model_config=None):

        super(BaseFE, self).__init__(
            name='fe', task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=1,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size
        )

        if optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.estimator_id = estimator_id

        path = 'FE-%s-%s(%d)_%s' % (
            self.estimator_id, optimizer, self.seed, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        _candidates = None
        if self.task_type in CLS_TASKS:
            _candidates = get_combined_candidtates(_classifiers, _cls_addons)
        else:
            _candidates = get_combined_candidtates(_regressors, _rgs_addons)

        self.cs = get_fe_cs(
            self.task_type, include_preprocessors=include_preprocessors, if_imbal=self.if_imbal
        )

        if model_config is None:
            if self.estimator_id in _candidates:
                rgs_class = _candidates[self.estimator_id]
            else:
                raise ValueError("Algorithm %s not supported!" % self.estimator_id)
            model_config = rgs_class.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
        elif isinstance(model_config, Configuration):
            model_config = model_config.get_dictionary()
        elif not isinstance(model_config, dict):
            raise ValueError("Invalid model_config type: %s" % str(type(model_config)))

        self.cs.add_hyperparameter(Constant('algorithm', estimator_id))
        for key, value in model_config.items():
            self.cs.add_hyperparameter(
                Constant("%s:%s" % (self.estimator_id, key), value)
            )

        # Define evaluator and optimizer
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.fe.fe_evaluator import FECLSEvaluator
            self.evaluator = FECLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal)
        else:
            from mindware.modules.fe.fe_evaluator import FERGSEvaluator
            self.evaluator = FERGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer('fe')

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-FE-%s-%s(%d)' % (self.estimator_id, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)
