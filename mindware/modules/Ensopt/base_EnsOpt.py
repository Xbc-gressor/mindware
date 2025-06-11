import os
import json
from typing import List, Union, Callable
from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import time
import datetime

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from .optimizer.myOptimizer import SMACEnsOptimizer
from mindware.components.utils.topk_saver import load_combined_transformer_estimator, CombinedTopKModelSaver

from mindware.components.feature_engineering.parse import construct_node
from .algorithm.ens_pooling import avging as ens_pooling
class BaseEnsOpt(BaseAutoML):
    
    name='EnsOpt_cash'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None, ens_size = 5, 
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test',
                 filter_params=None):

        super(BaseEnsOpt, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id
        )

        if optimizer not in ['smac', 'tpe', 'random_search', 'mab', 'block_0', 'block_1']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)
        self.ens_size = ens_size
        self.include_algorithms = include_algorithms
        path = 'CASH-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)

        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)

        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # Define evaluator and optimizer
        self.evaluator = None
        self.es = ens_pooling(ens_size)
        if self.task_type in CLS_TASKS:
            from .evaluator.ensOpt_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
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
            from .evaluator.ensOpt_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)
        self.optimizer = self.build_optimizer(name='cash', sub_optimizer=sub_optimizer)
        self.steps = 0
    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseEnsOpt, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf
    
    def iterate(self, trial_num=None):
        if trial_num is None:
            trial_num = self.inner_iter_num_per_iter

        self.optimizer.inner_iter_num_per_iter = trial_num

        self.optimizer.iterate(steps=self.steps, budget=self.time_limit + self.timestamp - time.time())
        if time.time() - self.timestamp > self.time_limit:
            self.timeout_flag = True
            if self.timeout_flag:
                self.logger.info(f"Time out({self.time_limit}s)!")
        self.early_stop_flag = self.optimizer.early_stopped_flag
        if self.early_stop_flag:
            self.logger.info(f"Early stop!")

        self.incumbent_perf = self.optimizer.incumbent_perf
        self.incumbent = self.optimizer.get_incumbent_config()
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def run(self, refit='full'):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()
                self.steps += 1

        return self.incumbent_perf
    
    def build_optimizer(self, name='hpo', **kwargs):
        optimizer = SMACEnsOptimizer(
            evaluator=self.evaluator, ens_size = self.ens_size, es = self.es, data_node = self.data_node, config_space=self.cs, name=name, eval_type=self.evaluation,
            time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
            per_run_time_limit=self.per_run_time_limit,
            inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
            output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk
        )

        return optimizer

    # def _predict(self, test_data: DataNode, refit='full'):
    #     try:
    #         best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.incumbent,
    #                                                                         self.datetime)
    #     except Exception as e:
    #         if self.evaluation == 'cv':
    #             raise AttributeError("Please call refit() for cross-validation!")
    #         else:
    #             raise e
    #     test_data_node = test_data.copy_()
    #     test_data_node = construct_node(test_data_node, best_op_list)

    #     if self.task_type in CLS_TASKS:
    #         return estimator.predict_proba(test_data_node.data[0])
    #     else:
    #         return estimator.predict(test_data_node.data[0])
        
    def _predict(self, test_data: DataNode, refit='full'):
        # 先不做refit(论文就是这样),其实只有CLS的
        preds = []
        for model in self.es.model_pool:
            if not model:
                continue
            if self.task_type in CLS_TASKS:
                pred = model.predict_proba(test_data.data[0])
            else:
                pred = model.predict(test_data.data[0])
            preds.append(pred)
        # 概率直接取平均
        return np.array(preds).mean(axis=0)

    def predict(self, test_data: DataNode, refit='full'):
        preds = self._predict(test_data, refit=refit)

        results = []
        for pred in preds:
            if self.task_type in CLS_TASKS:
                results.append(np.argmax(pred, axis=-1))
            else:
                results.append(pred)

        return results
    
    def get_conf(self, save=False):
        # 获取对象的配置信息
        conf = {
            'name': self.name,
            'task_type': self.task_type,
            'metric': self.metric_name,
            'evaluation': self.evaluation,
            'resampling_params': self.resampling_params,
            'optimizer': self.optimizer.name,
            'inner_iter_num_per_iter': self.inner_iter_num_per_iter,
            'time_limit': self.time_limit,
            'amount_of_resource': self.amount_of_resource,
            'per_run_time_limit': self.per_run_time_limit,
            'output_dir': self.output_dir,
            'seed': self.seed,
        }

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf
    def get_model_info(self, save=True):
        pass