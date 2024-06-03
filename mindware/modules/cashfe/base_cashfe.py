import os
import time
from typing import List

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.feature_engineering.task_space import get_task_hyperparameter_space

class BaseCASHFE(BaseAutoML):
    def __init__(self, include_algorithms: List[str] = None, sub_optimizer: str = 'smac',
                 metric: str = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac',
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 inner_iter_num_per_iter=1,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method=None, ensemble_size=5,
                 include_preprocessors=None):

        super(BaseCASHFE, self).__init__(
            name='cashfe',
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, per_run_time_limit=per_run_time_limit,
            time_limit=time_limit, amount_of_resource=amount_of_resource,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size
        )

        if optimizer not in ['smac', 'tpe', 'random_search', 'mab']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        path = 'CASHFE-%s(%d)_%s' % (
            optimizer, self.seed, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        self.cs = None
        if self.task_type in CLS_TASKS:
            from mindware.components.evaluators.cls_evaluator import get_cash_cs as get_cls_cash_cs
            self.cs = get_cls_cash_cs(self.include_algorithms, self.task_type)
        else:
            from mindware.components.evaluators.rgs_evaluator import get_cash_cs as get_rgs_cash_cs
            self.cs = get_rgs_cash_cs(self.include_algorithms, self.task_type)

        fe_config_space = get_task_hyperparameter_space(
            self.task_type, include_preprocessors=include_preprocessors, if_imbal=self.if_imbal
        )

        if self.optimizer_name != 'mab':
            self.cs.add_hyperparameters(fe_config_space.get_hyperparameters())
            self.cs.add_conditions(fe_config_space.get_conditions())
            self.cs.add_forbidden_clauses(fe_config_space.get_forbiddens())

        # Define evaluator and optimizer
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            # from mindware.modules.cash.cash_evaluator import CASHClassificationEvaluator
            # self.evaluator = CASHClassificationEvaluator(
            #     fixed_config=None,
            #     scorer=self.metric,
            #     data_node=data_node,
            #     if_imbal=self.if_imbal,
            #     timestamp=self.timestamp,
            #     output_dir=self.output_dir,
            #     seed=self.seed,
            #     resampling_strategy=evaluation,
            #     resampling_params=resampling_params)
            from mindware.modules.cashfe.cashfe_evaluator import CASHFECLSEvaluator
            self.evaluator = CASHFECLSEvaluator(
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
            # from mindware.modules.cash.cash_evaluator import CASHRegressionEvaluator
            # self.evaluator = CASHRegressionEvaluator(
            #     fixed_config=None,
            #     scorer=self.metric,
            #     data_node=data_node,
            #     timestamp=self.timestamp,
            #     output_dir=self.output_dir,
            #     seed=self.seed,
            #     resampling_strategy=evaluation,
            #     resampling_params=resampling_params)
            from mindware.modules.cashfe.cashfe_evaluator import CASHFERGSEvaluator
            self.evaluator = CASHFERGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer('cashfe', sub_optimizer=sub_optimizer, fe_config_space=fe_config_space)

        pass

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASHFE-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)
