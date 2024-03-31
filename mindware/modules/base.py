import os
import time
import datetime
import numpy as np
import pickle as pkl

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.metrics.metric import get_metric

from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.components.utils.constants import type_dict

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.utils.topk_saver import load_combined_transformer_estimator

from mindware.components.feature_engineering.parse import construct_node


class BaseAutoML(object):
    def __init__(self, task_type=None, metric: str = 'acc',
                 data_node: DataNode = None, evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', per_run_time_limit=600,
                 time_limit=600, amount_of_resource=None,
                 inner_iter_num_per_iter=1,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method=None, ensemble_size=5):

        self.metric = get_metric(metric)
        self.data_node = data_node
        self.evaluation = evaluation
        self.task_type = task_type
        self.seed = seed

        self.optimizer_name = optimizer
        self.per_run_time_limit = per_run_time_limit
        self.time_limit = time_limit
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource
        self.inner_iter_num_per_iter = inner_iter_num_per_iter

        self.timeout_flag = False
        self.early_stop_flag = False
        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.output_dir = output_dir
        self.n_jobs = n_jobs

        self.optimizer = None
        self.evaluator = None
        self.cs = None

        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.es = None

        self.incumbent_perf = -float("INF")
        self.incumbent = None
        self.eval_dict = dict()

        task_type = type_of_target(data_node.data[1])
        if task_type in type_dict:
            task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type

    def _get_logger(self, name):
        raise NotImplementedError()

    def build_optimizer(self, name='hpo'):

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

        optimizer = optimizer_class(self.evaluator, self.cs, name,
                                    eval_type=self.evaluation, output_dir=self.output_dir,
                                    time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
                                    per_run_time_limit=self.per_run_time_limit,
                                    inner_iter_num_per_iter=self.inner_iter_num_per_iter,
                                    timestamp=self.timestamp, seed=self.seed, n_jobs=self.n_jobs)

        return optimizer

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

        if self.ensemble_method is not None and self.evaluation in ['holdout', 'partial']:
            self.fit_ensemble()

        return self.incumbent_perf

    def fit_ensemble(self):
        if self.ensemble_method is not None:
            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      data_node=self.data_node,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=self.data_node)

    def predict(self, test_data: DataNode, ens=True):
        if self.task_type in CLS_TASKS:
            pred = self._predict(test_data, ens)
            return np.argmax(pred, axis=-1)
        else:
            return self._predict(test_data, ens)

    def _predict(self, test_data: DataNode, ens=True):
        if ens and self.ensemble_method is not None:
            if self.es is None and self.evaluation == 'cv':
                raise AttributeError("Please call refit() for cross-validation!")
            elif self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data)
        else:
            try:
                best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.incumbent,
                                                                              self.datetime)
            except Exception as e:
                if self.evaluation == 'cv':
                    raise AttributeError("Please call refit() for cross-validation!")
                else:
                    raise e
            test_data_node = test_data.copy_()
            test_data_node = construct_node(test_data_node, best_op_list)

            if self.task_type in CLS_TASKS:
                return estimator.predict_proba(test_data_node.data[0])
            else:
                return estimator.predict(test_data_node.data[0])

    def predict_proba(self, test_data: DataNode):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data)
