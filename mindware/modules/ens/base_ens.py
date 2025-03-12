import os
import time
import json
import datetime
import numpy as np
import pickle as pkl
import warnings

from typing import Union, Callable
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.metrics.metric import get_metric

from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.constants import type_dict

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.utils.logging_utils import setup_logger, get_logger


class BaseAutoML(object):
    def __init__(self, task_type: str = None, stats=None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac',
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir=None, seed=1, n_jobs=1, topk=50, rmfiles=False,
                 task_id='test'):

        if optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.name = 'ens'
        self.metric_name = 'unknown'
        if isinstance(metric, str):
            self.metric_name = metric
        self.metric = get_metric(metric)
        self.stats = stats
        self.data_node = data_node.copy_()
        self.evaluation = evaluation
        self.resampling_params = resampling_params
        self.seed = seed

        self.optimizer_name = optimizer
        self.per_run_time_limit = per_run_time_limit
        self.time_limit = time_limit
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource
        self.inner_iter_num_per_iter = 1

        self.timeout_flag = False
        self.early_stop_flag = False
        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.topk = topk
        self.rmfiles = rmfiles

        self.optimizer = None
        self.evaluator = None
        self.cs = None

        self.incumbent_perf = -float("INF")
        self.incumbent = None
        self.eval_dict = dict()

        if task_type is None:
            task_type = type_of_target(data_node.data[1])
            if task_type in type_dict:
                task_type = type_dict[task_type]
            else:
                raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type

        self.if_imbal = False
        if self.task_type in CLS_TASKS:
            self.if_imbal = is_imbalanced_dataset(self.data_node)

        self.refit_status = 'none'  # none, partial, full

        self.logger = None
        self.task_id = task_id

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-ENS-%s(%d)' % (optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def build_optimizer(self, name='ens', **kwargs):

        if self.evaluation == 'partial':
            optimizer_class = MfseOptimizer
        elif self.evaluation == 'partial_bohb':
            optimizer_class = BohbOptimizer
        else:
            # TODO: Support asynchronous BO
            if self.optimizer_name == 'random_search':
                optimizer_class = RandomSearchOptimizer
            elif self.optimizer_name == 'tpe':
                optimizer_class = TPEOptimizer
            elif self.optimizer_name == 'smac':
                optimizer_class = SMACOptimizer
            else:
                raise ValueError("Invalid optimizer %s" % self.optimizer_name)

        optimizer = optimizer_class(
            evaluator=self.evaluator, config_space=self.cs, name=name, eval_type=self.evaluation,
            time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
            per_run_time_limit=self.per_run_time_limit,
            inner_iter_num_per_iter=self.inner_iter_num_per_iter,timestamp=self.timestamp,
            output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk,
        )

        return optimizer

    def iterate(self, trial_num=None):
        if trial_num is None:
            trial_num = self.inner_iter_num_per_iter

        self.optimizer.inner_iter_num_per_iter = trial_num

        self.optimizer.iterate(budget=self.time_limit + self.timestamp - time.time())
        if time.time() - self.timestamp > self.time_limit:
            self.timeout_flag = True
        self.early_stop_flag = self.optimizer.early_stopped_flag

        self.incumbent_perf = self.optimizer.incumbent_perf
        self.incumbent = self.optimizer.incumbent_config
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def rm_files(self):
        self.logger.info('Start to delete files other than incumbent!')
        incumbent_id = CombinedTopKModelSaver.get_configuration_id(self.incumbent)
        for file in os.listdir(self.output_dir):
            if incumbent_id in file or file.endswith('.log') or file.endswith('.json') or file.endswith('topk_config.pkl'):
                continue
            os.remove(os.path.join(self.output_dir, file))

    def run(self):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        if self.rmfiles:
            self.rm_files()

        return self.incumbent_perf


    def predict_config(self, test_data: DataNode, config, refit=True):

        pred = self._predict_config(test_data, config=config, refit=refit)

        if self.task_type in CLS_TASKS:
            return np.argmax(pred, axis=-1)
        else:
            return pred

    def _predict_config(self, test_data: DataNode, config, refit=True):

        es = EnsembleBuilder(ensemble_method=config['ensemble_method'],
                             ensemble_size=config['ensemble_size'],
                             task_type=self.task_type,
                             metric=self.metric,
                             output_dir=self.output_dir, seed=self.seed)
        es.fit(stats=self.stats, datanode=self.data_node)

        if refit:
            es.refit(self.data_node)

        return es.predict(test_data, refit=refit)

    def predict_proba_config(self, test_data: DataNode, config, refit=True):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict_config(test_data, config, refit=refit)

