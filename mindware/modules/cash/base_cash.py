import os
import time
from typing import List

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.utils.functions import is_imbalanced_dataset


class BaseCASH(BaseAutoML):
    def __init__(self, include_algorithms: List[str] = None, task_type=None, metric: str = 'acc',
                 data_node: DataNode = None, evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', per_run_time_limit=600,
                 time_limit=600, amount_of_resource=None,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method=None, ensemble_size=5):

        super(BaseCASH, self).__init__(
            task_type=task_type, metric=metric,
            data_node=data_node, evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, per_run_time_limit=per_run_time_limit,
            time_limit=time_limit, amount_of_resource=amount_of_resource,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size
        )

        self.include_algorithms = include_algorithms
        path = 'CASH-%s(%d)_%s' % (
            optimizer, self.seed, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        self.cs = None
        if self.task_type in CLS_TASKS:
            from mindware.components.evaluators.cls_evaluator import get_cash_cs as get_cls_cash_cs
            self.if_imbal = is_imbalanced_dataset(self.data_node)
            self.cs = get_cls_cash_cs(self.include_algorithms, self.task_type)
        else:
            from mindware.components.evaluators.rgs_evaluator import get_cash_cs as get_rgs_cash_cs
            self.if_imbal = False
            self.cs = get_rgs_cash_cs(self.include_algorithms, self.task_type)

        # Define evaluator and optimizer
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHClassificationEvaluator
            self.evaluator = CASHClassificationEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                if_imbal=self.if_imbal,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=evaluation,
                resampling_params=resampling_params)
        else:
            from mindware.modules.cash.cash_evaluator import CASHRegressionEvaluator
            self.evaluator = CASHRegressionEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=evaluation,
                resampling_params=resampling_params)

        self.optimizer = self.build_optimizer('cash')

        pass

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)
