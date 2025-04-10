import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics._scorer import _BaseScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.components.ensemble.blending import Blending
from mindware.components.utils.constants import CLS_TASKS
from mindware.modules.base_evaluator import fetch_predict_estimator, fetch_predict_results
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import BaseEvaluator, get_kfold_name
from mindware.components.ensemble.parallel_fit import layer_fit


class Stacking(Blending):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 meta_learner='linear', stack_layers = 1, thread=16,
                 skip_connect=True, retain=True,
                 predictions=None, base_model_mask=None, opt=False):
        super().__init__(stats=stats,
                ensemble_size=ensemble_size,
                task_type=task_type, if_imbal=if_imbal,
                metric=metric, resampling_params=resampling_params,
                output_dir=output_dir, seed=seed,
                meta_learner=meta_learner, stack_layers=stack_layers, thread=thread,
                skip_connect=skip_connect, retain=retain,
                predictions=predictions, base_model_mask=base_model_mask)

        self.ensemble_method = "stacking"
        self.folds = 3
        if self.resampling_params is not None and 'folds' in self.resampling_params:
            self.folds = self.resampling_params['folds']
        self.folds = self.sfolds

        self.opt = opt
        self.base_sms = None
        self.base_ops = None

    def get_base_features(self, datanode, val_nodes: dict=None, mode='partial'):
        _output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
        model_cnt = 0
        suc_cnt = 0
        ori_xs = {'train': None}

        stack_configs = [[], []]
        ori_config_paths = []
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    stack_configs[0].append(config)
                    ori_config_paths.append(path)
                    if self.skip_connect:
                        if ori_xs['train'] is None: ori_xs = {'train': []}
                        model_path = path if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(path, 'full')
                        op_list, _, _ = CombinedTopKModelSaver._load(model_path)
                        ori_x = construct_node(datanode.copy_(), op_list).data[0]
                        ori_xs['train'].append(ori_x)
                        if val_nodes is not None:
                            for key in val_nodes.keys():
                                if key not in ori_xs: ori_xs[key] = []
                                ori_x = construct_node(val_nodes[key].copy_(), op_list).data[0]
                                ori_xs[key].append(ori_x)

                    suc_cnt += 1
                model_cnt += 1

        sms, ops, base_feature, _, cost = layer_fit(stack_configs=stack_configs, new_node=datanode, ori_xs=None, n_base_model=len(stack_configs[0]),
                                            task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed, 
                                            layer=0, thread=self.thread, folds=self.folds, output_dir=_output_dir, ori_config_paths=ori_config_paths, logger=self.logger, mode=mode)

        self.base_sms = sms
        self.base_ops = ops

        n_dim = base_feature.shape[1] // self.ensemble_size

        base_features = {'train': base_feature}
        if val_nodes is not None:
            for key in val_nodes.keys():
                base_features[key] = np.zeros((val_nodes[key].data[0].shape[0], base_feature.shape[1]))
                for suc_cnt, config in enumerate(stack_configs[0]):
                    # path = ori_config_paths[suc_cnt] if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(ori_config_paths[suc_cnt], 'full')
                    # op_list, estimator, _ = CombinedTopKModelSaver._load(model_path)
                    # pred = fetch_predict_results(self.task_type, op_list, estimator, val_nodes[key])
                    # if len(pred.shape) == 1:
                    #     pred = pred.reshape(-1, 1)
                    # base_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:]
                    estimators = sms[suc_cnt]
                    op_lists = ops[suc_cnt]
                    for estimator, op_list in zip(estimators, op_lists):
                        _new_node = val_nodes[key]
                        pred = fetch_predict_results(self.task_type, op_list, estimator, _new_node)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                        base_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)

        self.logger.info(f"Cost of Layer0 training with {self.thread} threads: {cost}s")
        self.train_cost.append(cost)

        return base_features, ori_xs

    def get_feature(self, datanode, mode):
        # Predict the labels via blending
        base_features = None
        ori_xs = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    estimators = self.base_sms[suc_cnt]
                    op_lists = self.base_ops[suc_cnt]
                    for estimator, op_list in zip(estimators, op_lists):
                        pred = fetch_predict_results(self.task_type, op_list, estimator, datanode)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                        n_dim = pred.shape[1] if pred.shape[1] > 2 else 1
                        if base_features is None:
                            num_samples = len(datanode.data[0])
                            base_features = np.zeros((num_samples, self.ensemble_size * n_dim))
                        base_features[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)
                    if self.skip_connect:
                        if ori_xs is None: ori_xs = []
                        model_path = path if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(path, 'full')
                        op_list, _, _ = CombinedTopKModelSaver._load(model_path)
                        ori_x = construct_node(datanode.copy_(), op_list).data[0]
                        ori_xs.append(ori_x)
                    suc_cnt += 1

                model_cnt += 1

        return base_features, ori_xs

    def get_ens_model_info(self):
        ens_info = super().get_ens_model_info()
        ens_info['ensemble_method'] = 'stacking'
        ens_info['folds'] = [self.folds, self.sfolds]
        return ens_info

    def refit(self, datanode, mode):
        super().refit(datanode, mode)
        if self.opt:
            # Train basic models using a part of training data
            base_features, ori_xs = self.get_base_features(datanode, mode=mode)

            final_labels = {'train': datanode.data[1]}
            last_features = self.forward(base_features, final_labels, train=True, ori_xs=ori_xs)

            self.meta_learner = self.build_meta_learner(self.meta_method, self.task_type, last_features['train'], final_labels['train'],
                                                        ensemble_size=self.ensemble_size, if_imbal=self.if_imbal, metric=self.metric)
