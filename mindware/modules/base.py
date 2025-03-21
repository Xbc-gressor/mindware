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
from mindware.components.optimizers.mab_optimizer import MabOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.constants import type_dict

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.utils.topk_saver import load_combined_transformer_estimator, CombinedTopKModelSaver

from mindware.components.feature_engineering.parse import construct_node
from mindware.components.ensemble import ensemble_list

from mindware.components.evaluators.base_evaluator import fetch_predict_estimator
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.feature_engineering.parse import parse_config
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator


class BaseAutoML(object):
    name = 'abstract'

    def __init__(self, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test'):

        self.metric_name = 'unknown'
        if isinstance(metric, str):
            self.metric_name = metric
        self.metric = get_metric(metric)
        self.data_node = data_node.copy_()
        self.evaluation = evaluation
        self.resampling_params = resampling_params
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
        self.topk = topk
        self.rmfiles = rmfiles

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

        self.refit_status = 'none'  # none, partial, full

        self.logger = None
        self.task_id = task_id

    @staticmethod
    def _recommand_models(task_type, task_id, data_node, metric, n_algo, include_algorithms=None):

        from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch_weight import \
            RankNetAdvisor as RankNetAdvisor_W
        alad = RankNetAdvisor_W(task_type=task_type, metric=metric)  # exclude_datasets=[task_id]
        alad.fit()

        model_candidates = alad.fetch_algorithm_set(task_id, datanode=data_node)
        include_models = list()
        for algo in model_candidates:
            if (include_algorithms is None or algo in include_algorithms) and len(include_models) < n_algo:
                include_models.append(algo)

        return include_models

    @staticmethod
    def _recommand_preps(task_type, task_id, data_node, metric, n_prep, include_algorithms, include_preprocessors=None):

        from mindware.components.meta_learning.fe_recomendation.gbm_advisor import GBMAdvisor
        alad_g = GBMAdvisor(task_type=task_type, metric=metric,
                            include_algorithms=include_algorithms)  # exclude_datasets=[task_id],
        alad_g.fit(save_flag=True)

        preprocessors = alad_g.fetch_preprocessor_set(task_id, datanode=data_node)
        include_preps = dict()
        for algo in preprocessors:
            tmp_prep = list()
            for prep in preprocessors[algo]:
                if (include_preprocessors is None or prep in include_preprocessors) and len(tmp_prep) < n_prep:
                    tmp_prep.append(prep)
            tmp_prep = ['empty'] + sorted(tmp_prep)
            include_preps[algo] = tmp_prep

        return include_preps

    @staticmethod
    def _get_valid_data(task_type, data_node, resampling_params=None, seed=1, train=False):

        test_size = 0.33
        if resampling_params is not None and 'test_size' in resampling_params:
            test_size = resampling_params['test_size']

        valid_data = data_node.copy_(no_data=True)
        X, y = data_node.data
        if task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=seed)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=seed)

        x_p2, y_p2 = None, None
        for train_index, val_index in ss.split(X, y):
            if train:
                val_index = train_index
            x_p2, y_p2 = X[val_index], y[val_index]

        valid_data.data = [x_p2, y_p2]

        return valid_data

    @classmethod
    def _refit_config(cls, config, data_node, task_type, if_imbal=False):
        algo_id = config['algorithm']
        if cls.name in ['fe', 'cashfe']:
            data_node, op_list = parse_config(data_node, config, record=True, if_imbal=if_imbal)
        else:
            op_list = {}

        estimator = fetch_predict_estimator(task_type, algo_id, config,
                                            data_node.data[0], data_node.data[1],
                                            weight_balance=data_node.enable_balance,
                                            data_balance=data_node.data_balance)

        return op_list, estimator

    def _get_logger(self, optimizer_name):
        raise NotImplementedError()

    def build_optimizer(self, name='hpo', **kwargs):

        if self.optimizer_name.startswith("block"):
            tree_id = int(self.optimizer_name.split('_')[1])
            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_execution_tree, \
                get_opt_node_type
            tree = get_opt_execution_tree(tree_id)

            if self.evaluation == 'partial':
                sub_optimizer = 'mfse'
            elif self.evaluation == 'partial_bohb':
                sub_optimizer = 'bohb'
            else:
                sub_optimizer = kwargs.get('sub_optimizer', 'smac')

            fe_config_space_dict = kwargs.get('fe_config_space_dict', None)

            optimizer = get_opt_node_type(tree, 0)(
                node_list=tree, node_index=0,
                evaluator=self.evaluator, cash_config_space=self.cs, name=name, eval_type=self.evaluation,
                time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
                per_run_time_limit=self.per_run_time_limit,
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                sub_optimizer=sub_optimizer, fe_config_space_dict=fe_config_space_dict,
                output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk
            )

            return optimizer

        opt_paras = {}
        if self.optimizer_name == 'mab':
            optimizer_class = MabOptimizer
            if self.evaluation == 'partial':
                opt_paras['sub_optimizer'] = 'mfse'
            elif self.evaluation == 'partial_bohb':
                opt_paras['sub_optimizer'] = 'bohb'
            else:
                opt_paras['sub_optimizer'] = kwargs.get('sub_optimizer', 'smac')
            opt_paras['fe_config_space'] = kwargs.get('fe_config_space', None)

        elif self.evaluation == 'partial':
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
            inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
            output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk,
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
        self.incumbent = self.optimizer.incumbent_config
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def rm_files(self):
        self.logger.info('Start to delete files other than incumbent!')
        incumbent_id = CombinedTopKModelSaver.get_configuration_id(self.incumbent)
        for file in os.listdir(self.output_dir):
            if incumbent_id in file or file.endswith('.log') or file.endswith('.json') or file.endswith(
                    'topk_config.pkl'):
                continue
            os.remove(os.path.join(self.output_dir, file))

    def run(self, refit=True):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        if self.ensemble_method is not None:
            if self.evaluation == 'cv':
                self.refit(partial=True)

            self.fit_ensemble(refit)  # 如果是cv，就不再refit

        if self.evaluation == 'cv' or refit:
            if self.refit_status != 'full':
                self.refit_incumbent()

        if self.rmfiles:
            self.rm_files()

        return self.incumbent_perf

    def refit_incumbent(self):

        self.logger.debug('Start to refit the best model!')

        if self.incumbent is None:
            raise AssertionError("The best config is None! Please check if all the evaluations are failed!")

        model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime,
                                                               refit=True)
        if os.path.exists(model_path):
            self.logger.debug("The best model has been refitted!")
            return

        config = self.incumbent.copy()
        algo_id = config['algorithm']
        if algo_id != 'neural_network':
            op_list, estimator = self._refit_config(self.incumbent, self.data_node, task_type=self.task_type,
                                                    if_imbal=self.if_imbal)
            CombinedTopKModelSaver._save([op_list, estimator, self.incumbent_perf], model_path)

    # train with whole data
    def refit(self, partial=False):
        # if partial, holdout training; else, whole-data training

        if self.ensemble_method is None:
            self.logger.error("No ensemble method is specified, no need to refit!")

        self.logger.debug('Start to refit all the well-performed models!')
        config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)

        if not os.path.exists(config_path):
            warnings.warn("Config path %s not found! Please check if all the evaluations are failed!" % config_path)
            return

        with open(config_path, 'rb') as f:
            stats = pkl.load(f)
        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        for algo_id in stats.keys():
            if algo_id == 'neural_network':
                continue
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                if not partial:
                    path = CombinedTopKModelSaver.get_refit_path(path)
                if os.path.exists(path):
                    continue
                # TODO: 有的refit会报错，提示X有NaN。原来的X是没有NaN的，可能FE后用一部分数据的时候没有NaN，但是全数据里面有了。
                try:
                    train_node = self.data_node
                    X, y = train_node.data[0], train_node.data[1]
                    if partial:
                        train_node = train_node.copy_(no_data=True)
                        ss = self.evaluator._get_spliter('holdout', test_size=test_size, random_state=self.seed)
                        for train_index, _ in ss.split(X, y):
                            X, y = X[train_index], y[train_index]

                        train_node.data = [X, y]

                    op_list, estimator = self._refit_config(config, data_node=train_node, task_type=self.task_type,
                                                            if_imbal=self.if_imbal)

                    CombinedTopKModelSaver._save([op_list, estimator, perf], path)
                except:
                    self.logger.error("Failed to refit for %s !" % path)

        self.refit_status = 'partial' if partial else 'full'

    def fit_ensemble(self, refit=True):

        self.logger.debug('Start to fit ensemble model!')

        if self.ensemble_method is not None:

            # 如果用全数据refit了，就不能包含k_nearest_neighbors, 因为它会将训练数据都预测为label，selection算法只会选knn
            if self.evaluation == 'cv':
                if self.refit_status in ['none', 'full']:
                    raise AttributeError("Please call refit(partial=True) for cross-validation!")

            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)

            valid_data = self._get_valid_data(task_type=self.task_type, data_node=self.data_node,
                                              resampling_params=self.resampling_params, seed=self.seed)
            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats, valid_data=valid_data,
                                      task_type=self.task_type, if_imbal=self.if_imbal,
                                      metric=self.metric,
                                      output_dir=self.output_dir, seed=self.seed)
            self.es.build_ensemble(ensemble_method=self.ensemble_method, ensemble_size=self.ensemble_size)
            self.es.fit()

            if refit and self.refit_status != 'full':
                self.es.refit(datanode=self.data_node)
        else:
            raise ValueError("No ensemble method is specified!")

    def predict(self, test_data: DataNode, refit=True, ens=True):
        pred = self._predict(test_data, refit=refit, ens=ens)

        if self.task_type in CLS_TASKS:
            return np.argmax(pred, axis=-1)
        else:
            return pred

    @classmethod
    def _fix_model(cls, task_type, data_node: DataNode = None, stats=None, resampling_params=None, seed=1,
                   if_imbal=False):
        metric = get_metric('mse')
        for algo_id in stats.keys():
            if algo_id not in ['xgboost']:
                continue
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                # _, _, perf = CombinedTopKModelSaver._load(path)
                train_node = cls._get_valid_data(task_type=task_type, data_node=data_node,
                                                 resampling_params=resampling_params, seed=seed, train=True)
                valid_node = cls._get_valid_data(task_type=task_type, data_node=data_node,
                                                 resampling_params=resampling_params, seed=seed)

                op_list, estimator = cls._refit_config(config, data_node=train_node, task_type=task_type,
                                                       if_imbal=if_imbal)

                valid_node = construct_node(valid_node, op_list)

                pred = estimator.predict(valid_node.data[0])
                new_perf = metric._score_func(pred, valid_node.data[1]) * metric._sign

                CombinedTopKModelSaver._save([op_list, estimator, new_perf], path)

    @classmethod
    def _predict_stats(cls, task_type, metric: Union[str, Callable, _BaseScorer] = None, data_node: DataNode = None,
                       test_data: DataNode = None, stats=None,
                       resampling_params=None,
                       ensemble_method=None, ensemble_size=None, refit=True, prob=False, output_dir='./data', seed=1,
                       task_id='test'):
        path = 'STA-(%d)-%s_%s' % (
            seed, task_id, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        )
        output_dir = os.path.join(output_dir, path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Predicting with stats")
        logger_name = 'MindWare-STA-(%d)' % (seed)
        setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))
        logger = get_logger(logger_name)

        if_imbal = False
        if task_type in CLS_TASKS:
            if_imbal = is_imbalanced_dataset(data_node)

        # cls._fix_model(task_type=task_type, data_node=data_node, stats=stats, resampling_params=resampling_params, seed=seed, if_imbal=if_imbal)

        stats = stats.copy()
        metric_name = 'unknown'
        if isinstance(metric, str):
            metric_name = metric
        metric = get_metric(metric)

        best_path = None
        best_config = None
        best_perf = -float("INF")
        for algo_id in stats.keys():
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                if perf > best_perf:
                    best_perf = perf
                    best_config = config
                    best_path = path

        if ensemble_method is not None:
            valid_data = cls._get_valid_data(task_type=task_type, data_node=data_node,
                                             resampling_params=resampling_params, seed=seed)
            es = EnsembleBuilder(stats=stats, valid_data=valid_data,
                                 task_type=task_type, if_imbal=if_imbal,
                                 metric=metric,
                                 output_dir=output_dir, seed=seed)
            es.build_ensemble(ensemble_method=ensemble_method, ensemble_size=ensemble_size, **resampling_params)
            es.fit()
            if refit:
                es.refit(datanode=data_node)
            pred = es.predict(test_data, refit)
            if task_type in CLS_TASKS:
                if prob:
                    return pred
                else:
                    return np.argmax(pred, axis=-1)
            else:
                return pred

        else:

            if best_path is None:
                raise AttributeError("No stats found!")

            if refit:
                logger.info('Start to refit the best model!')
                best_path = CombinedTopKModelSaver.get_refit_path(best_path)
                if os.path.exists(best_path):
                    logger.info("The best model has been refitted!")
                    best_op_list, estimator, _ = CombinedTopKModelSaver._load(best_path)
                else:
                    best_op_list, estimator = cls._refit_config(best_config, data_node, task_type=task_type,
                                                                if_imbal=if_imbal)
            else:
                best_op_list, estimator, _ = CombinedTopKModelSaver._load(best_path)
            test_data_node = test_data.copy_()
            test_data_node = construct_node(test_data_node, best_op_list)

            conf = {
                'name': cls.name,
                'task_type': task_type,
                'task_id': task_id,
                'metric': metric_name,
                'seed': seed,
                'if_imbal': if_imbal,
                'ensemble_method': ensemble_method,
                'ensemble_size': ensemble_size
            }
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

            if task_type in CLS_TASKS:
                if prob:
                    return estimator.predict_proba(test_data_node.data[0])
                else:
                    return np.argmax(estimator.predict_proba(test_data_node.data[0]), axis=-1)
            else:
                return estimator.predict(test_data_node.data[0])

    def _predict(self, test_data: DataNode, refit=True, ens=True):
        if ens and self.ensemble_method is not None:
            if self.es is None and self.evaluation == 'cv':
                raise AttributeError("Please call refit() for cross-validation!")
            elif self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data, refit)
        else:
            try:
                best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.incumbent,
                                                                              self.datetime, refit=refit)
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

    def predict_proba(self, test_data: DataNode, refit=True, ens=True):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data, refit=refit, ens=ens)

    def get_model_info(self, save=False):
        model_info = dict()
        if self.es is not None:
            model_info['ensemble'] = self.es.get_ens_model_info()
        path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime)
        model_info['best'] = (self.incumbent['algorithm'], self.incumbent, path)

        opt_trajectory = self.optimizer.get_opt_trajectory()
        if opt_trajectory is not None:
            model_info['opt_trajectory'] = opt_trajectory

        if save:
            with open(os.path.join(self.output_dir, 'best_model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=4)

        return model_info

    def get_conf(self):
        # 获取对象的配置信息
        conf = {
            'name': self.name,
            'task_type': self.task_type,
            'task_id': self.task_id,
            'metric': self.metric_name,
            'optimizer': self.optimizer_name,
            'time_limit': self.time_limit,
            'amount_of_resource': self.amount_of_resource,
            'per_run_time_limit': self.per_run_time_limit,
            'evaluation': self.evaluation,
            'seed': self.seed,
            'if_imbal': self.if_imbal,
            'ensemble_method': self.ensemble_method,
            'ensemble_size': self.ensemble_size
        }
        if hasattr(self, 'cs_args'):
            conf['cs_args'] = self.cs_args
            for key in conf['cs_args']:
                if isinstance(conf['cs_args'][key], np.bool_):
                    conf['cs_args'][key] = bool(conf['cs_args'][key])

        if hasattr(self, 'filter_params'):
            conf['filter_params'] = self.filter_params

        return conf
