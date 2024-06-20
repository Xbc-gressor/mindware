import os
import time
import datetime
import numpy as np
import pickle as pkl
import warnings

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.metrics.metric import get_metric

from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer
from mindware.components.optimizers.mab_optimizer import MabOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.constants import type_dict

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.utils.topk_saver import load_combined_transformer_estimator

from mindware.components.feature_engineering.parse import construct_node
from mindware.components.ensemble import ensemble_list


class BaseAutoML(object):
    def __init__(self, name: str, task_type: str = None,
                 metric: str = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method=None, ensemble_size=5):

        self.name = name

        self.metric = get_metric(metric)
        self.data_node = data_node.copy_()
        self.evaluation = evaluation
        self.resampling_params = resampling_params
        self.seed = seed

        self.optimizer_name = optimizer
        self.per_run_time_limit = per_run_time_limit
        self.time_limit = time_limit
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource
        if self.optimizer_name != 'mab':
            inner_iter_num_per_iter = 1
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

        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.es = None

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

        self.already_refit = False

        self.logger = None

    def _get_logger(self, name):
        raise NotImplementedError()

    def build_optimizer(self, name='hpo', **kwargs):

        opt_paras = {}
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
            elif self.optimizer_name == 'mab':
                optimizer_class = MabOptimizer
                opt_paras['sub_optimizer'] = kwargs.get('sub_optimizer', 'smac')
                opt_paras['fe_config_space'] = kwargs.get('fe_config_space', None)
            else:
                raise ValueError("Invalid optimizer %s" % self.optimizer_name)

        optimizer = optimizer_class(
            evaluator=self.evaluator, config_space=self.cs, name=name,
            eval_type=self.evaluation,
            time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
            per_run_time_limit=self.per_run_time_limit,
            output_dir=self.output_dir, timestamp=self.timestamp,
            inner_iter_num_per_iter=self.inner_iter_num_per_iter,
            seed=self.seed, n_jobs=self.n_jobs,
            **opt_paras
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
        self.incumbent = self.optimizer.incumbent_config.get_dictionary().copy()
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def run(self, refit=True):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        if refit:
            self.refit()
        else: # refit() 中会fit_ensemble
            if self.ensemble_method is not None:
                if self.evaluation in ['holdout', 'partial', 'partial_bohb']:
                    self.fit_ensemble()

        return self.incumbent_perf

    # train with whole data
    def refit(self):
        from mindware.components.evaluators.base_evaluator import fetch_predict_estimator
        from mindware.components.utils.topk_saver import CombinedTopKModelSaver
        from mindware.components.feature_engineering.parse import parse_config

        if self.ensemble_method is not None:
            self.logger.info('Start to refit all the well-performed models!')
            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)

            if not os.path.exists(config_path):
                warnings.warn("Config path %s not found! Please check if all the evaluations are failed!" % config_path)
                return

            with open(config_path, 'rb') as f:
                stats = pkl.load(f)
            for algo_id in stats.keys():
                model_to_eval = stats[algo_id]
                for idx, (config, perf, path) in enumerate(model_to_eval):
                    
                    # TODO: 有的refit会报错，提示X有NaN。原来的X是没有NaN的，可能FE后用一部分数据的时候没有NaN，但是全数据里面有了。
                    try:
                        if self.name in ['fe', 'cashfe']:
                            data_node, op_list = parse_config(self.data_node.copy_(), config, record=True,
                                                            if_imbal=self.if_imbal)
                        else:
                            op_list = {}
                            data_node = self.data_node.copy_()

                        algo_id = config['algorithm']
                        estimator = fetch_predict_estimator(self.task_type, algo_id, config,
                                                            data_node.data[0], data_node.data[1],
                                                            weight_balance=data_node.enable_balance,
                                                            data_balance=data_node.data_balance)
                        with open(path, 'wb') as f:
                            pkl.dump([op_list, estimator, None], f)
                    except:
                        self.logger.error("Failed to refit for %s !" % path)

            self.fit_ensemble()

        else:
            self.logger.info('Start to refit the best model!')

            if self.incumbent is None:
                warnings.warn("The best config is None! Please check if all the evaluations are failed!")
                return

            model_path = os.path.join(self.output_dir, '%s_%s.pkl' % (
                self.timestamp, CombinedTopKModelSaver.get_configuration_id(self.incumbent)))
            config = self.incumbent.copy()

            if self.name in ['fe', 'cashfe']:
                data_node, op_list = parse_config(self.data_node.copy_(), config, record=True,
                                                  if_imbal=self.if_imbal)
            else:
                op_list = {}
                data_node = self.data_node.copy_()

            algo_id = config['algorithm']
            estimator = fetch_predict_estimator(self.task_type, algo_id, config,
                                                data_node.data[0], data_node.data[1],
                                                weight_balance=data_node.enable_balance,
                                                data_balance=data_node.data_balance)
            with open(model_path, 'wb') as f:
                pkl.dump([op_list, estimator, None], f)

        self.already_refit = True

    def fit_ensemble(self):
        if self.ensemble_method is not None:
            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)

            # 如果用全数据refit了，就不能包含k_nearest_neighbors, 因为它会将训练数据都预测为label，selection算法只会选knn
            if self.already_refit and self.ensemble_method == 'ensemble_selection':
                stats.remove('k_nearest_neighbors')

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      data_node=self.data_node,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=self.data_node)

    def predict(self, test_data: DataNode, ens=True, prob=False):
        pred = self._predict(test_data, ens)

        if self.task_type in CLS_TASKS:
            if prob:
                return pred
            else:
                return np.argmax(pred, axis=-1)
        else:
            return pred

    def _predict_stats(self, test_data: DataNode, stats, ens=False, prob=False):
        # 如果用全数据refit了，就不能包含k_nearest_neighbors, 因为它会将训练数据都预测为label，selection算法只会选knn
        if self.already_refit and self.ensemble_method == 'ensemble_selection':
            stats.remove('k_nearest_neighbors')

        print("Predicting with stats")
        if ens and self.ensemble_method is not None:
            es = EnsembleBuilder(stats=stats,
                                 data_node=self.data_node,
                                 ensemble_method=self.ensemble_method,
                                 ensemble_size=self.ensemble_size,
                                 task_type=self.task_type,
                                 metric=self.metric,
                                 output_dir=self.output_dir)
            es.fit(data=self.data_node)
            pred = es.predict(test_data)
            if self.task_type in CLS_TASKS:
                if prob:
                    return pred
                else:
                    return np.argmax(pred, axis=-1)
            else:
                return pred

        else:
            best_path = None
            best_perf = -float("INF")
            for algo_id in stats.keys():
                model_to_eval = stats[algo_id]
                for idx, (config, perf, path) in enumerate(model_to_eval):
                    if perf > best_perf:
                        best_perf = perf
                        best_path = path

            if best_path is None:
                raise AttributeError("No stats found!")

            with open(best_path, 'rb') as f:
                best_op_list, estimator, _ = pkl.load(f)
            test_data_node = test_data.copy_()
            test_data_node = construct_node(test_data_node, best_op_list)

            if self.task_type in CLS_TASKS:
                if prob:
                    return estimator.predict_proba(test_data_node.data[0])
                else:
                    return np.argmax(estimator.predict_proba(test_data_node.data[0]), axis=-1)
            else:
                return estimator.predict(test_data_node.data[0])

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
