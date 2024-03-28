import os
import time
from typing import List

from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.components.utils.constants import type_dict


class BaseCASHOptimizer(object):
    def __init__(self, include_algorithms: List[str] = None, task_type=None, scorer: str = 'acc',
                 data_node: DataNode = None, evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', per_run_time_limit=600,
                 time_limit=600, amount_of_resource=None,
                 output_dir=None, seed=None, n_jobs=1):
        self.include_algorithms = include_algorithms
        self.scorer = scorer
        self.data_node = data_node
        self.task_type = task_type
        self.seed = seed

        self.time_limit = time_limit
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource

        self.timeout_flag = False
        self.early_stop_flag = False
        self.timestamp = time.time()

        self.output_dir = output_dir
        self.logger = self._get_logger(optimizer)

        self.incumbent_perf = -float("INF")
        self.incumbent = None
        self.eval_dict = dict()

        task_type = type_of_target(data_node.data[1])
        if task_type in type_dict:
            task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type

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
                scorer=self.scorer,
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
                scorer=self.scorer,
                data_node=data_node,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=evaluation,
                resampling_params=resampling_params)

        if evaluation == 'partial':
            optimizer_class = MfseOptimizer
        elif evaluation == 'partial_bohb':
            optimizer_class = BohbOptimizer
        else:
            # TODO: Support asynchronous BO
            if optimizer == 'random_search':
                optimizer_class = RandomSearchOptimizer
            elif optimizer == 'tpe':
                optimizer_class = TPEOptimizer
            elif optimizer == 'smac':
                optimizer_class = SMACOptimizer
            else:
                raise ValueError("Invalid optimizer %s" % optimizer)

        self.optimizer = optimizer_class(self.evaluator, self.cs, 'cash',
                                         eval_type=evaluation, output_dir=self.output_dir,
                                         time_limit=time_limit, evaluation_limit=self.amount_of_resource,
                                         per_run_time_limit=per_run_time_limit,
                                         inner_iter_num_per_iter=1,
                                         timestamp=self.timestamp, seed=self.seed, n_jobs=n_jobs)

        pass

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def iterate(self):
        self.optimizer.iterate(budget=self.time_limit + self.timestamp - time.time())
        if time.time() - self.timestamp > self.time_limit:
            self.timeout_flag = True
        self.early_stop_flag = self.optimizer.early_stopped_flag

        self.incumbent_perf = self.optimizer.incumbent_perf
        self.incumbent = self.optimizer.incumbent_config.get_dictionary().copy()
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def run(self):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        return self.incumbent_perf
