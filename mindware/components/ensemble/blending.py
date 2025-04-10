import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.ensemble.parallel_fit import layer_fit, save_ori_x, rm_ori_x
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
from mindware.modules.base_evaluator import fetch_predict_results
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.utils.data_manager import DataManager
from mindware.modules.base_evaluator import BaseEvaluator
from sklearn.metrics._scorer import _BaseScorer, _PredictScorer, _ThresholdScorer
from copy import deepcopy
from mindware.modules.ens.ens_utils import better_ens


class Besting:
    def __init__(self, task_type, best_idx, n_dim, ensemble_size):

        self.task_type = task_type
        self.best_idx = best_idx
        self.n_dim = n_dim
        self.ensemble_size = ensemble_size

    def stack_predict(self, features):

        # features shape: num_data, n_base_model*n_dim
        assert features.shape[1] == self.ensemble_size * self.n_dim

        pred = features[:, self.best_idx * self.n_dim:(self.best_idx + 1) * self.n_dim]

        if self.task_type in CLS_TASKS and self.n_dim == 1:
            pred = np.hstack([1-pred, pred])

        if pred.shape[1] == 1:
            pred = pred.reshape(-1)

        return pred

class Blending(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 meta_learner='auto', stack_layers = 1, thread=1,
                 skip_connect=True, retain=False,
                 predictions=None, base_model_mask=None):
        super().__init__(stats,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type, if_imbal=if_imbal,
                         metric=metric, resampling_params=resampling_params,
                         output_dir=output_dir, seed=seed,
                         predictions=predictions)

        self.thread = thread
        if self.resampling_params is not None and 'thread' in self.resampling_params:
            self.thread = self.resampling_params['thread']

        self.stack_layers = stack_layers
        if self.resampling_params is not None and 'stack_layers' in self.resampling_params:
            self.stack_layers = self.resampling_params['stack_layers']

        self.base_model_mask = base_model_mask
        self.stack_models = None
        self.layer_loss = []
        self.train_cost = []
        self.skip_connect = skip_connect
        self.retain = retain
        self.encoder = OneHotEncoder()

        self.leader_board = {'train': {}}

        if meta_learner == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
            except:
                warnings.warn("Lightgbm is not imported! Blending will use linear model instead!")
                meta_learner = 'linear'
        self.meta_method = meta_learner
        self.sfolds = 5
        self.lock = False

        self.best_config = None

    @staticmethod
    def build_meta_learner(meta_method, task_type, last_features, final_label=None, **kwargs):

        if meta_method in ['weighted', 'equal']:
            ensemble_size = kwargs.get('ensemble_size', None)
            from mindware.components.ensemble.ensemble_selection import EnsembleSelection
            data_len = last_features.shape[0]
            n_dim = last_features.shape[1] // ensemble_size
            last_features = last_features.reshape(data_len, -1, n_dim).transpose(1, 0, 2)
            if task_type in CLS_TASKS and n_dim == 1:
                last_features = np.concatenate([1-last_features, last_features], axis=2)
            meta_learner = EnsembleSelection(stats=None,
                                            ensemble_size=ensemble_size,
                                            task_type=task_type, if_imbal=kwargs['if_imbal'],
                                            metric=kwargs['metric'],
                                            predictions=last_features)
            if meta_method == 'weighted':
                meta_learner.fit(final_label)
            else:
                meta_learner.equal_fit()
        elif meta_method.startswith('best_'):
            ensemble_size = kwargs.get('ensemble_size', None)
            n_dim = last_features.shape[1] // ensemble_size
            best_idx = int(meta_method.split('_')[1][3:])
            meta_learner = Besting(task_type=task_type, best_idx=best_idx, n_dim=n_dim, ensemble_size=ensemble_size)
        else:
            # We use Xgboost as default meta-learner
            if task_type in CLS_TASKS:
                if meta_method == 'linear':
                    try:
                        from sklearn.linear_model import LogisticRegression
                    except:
                        from sklearn.linear_model.logistic import LogisticRegression
                    meta_learner = LogisticRegression(max_iter=1000)
                elif meta_method == 'gb':
                    try:
                        from sklearn.ensemble import GradientBoostingClassifier
                    except:
                        from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

                    meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                                n_estimators=250)
                elif meta_method == 'lightgbm':
                    from lightgbm import LGBMClassifier
                    meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150, n_jobs=1, verbose=-1)
            else:
                if meta_method == 'linear':
                    from sklearn.linear_model import LinearRegression
                    meta_learner = LinearRegression()
                elif meta_method == 'lightgbm':
                    from lightgbm import LGBMRegressor
                    meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70, n_jobs=1)

            meta_learner.fit(last_features, final_label)

        return meta_learner

    def get_base_features(self, datanode, val_nodes: dict=None):

        base_features, ori_xs = self.get_feature(datanode, mode='partial')

        return {'train': base_features}, {'train': ori_xs}

    def cal_scores(self, last_features, final_labels, n_base_model):
        n_dim = None
        losses = {}
        for key in last_features.keys():
            if n_dim is None:
                n_dim = last_features[key].shape[1] // n_base_model

            _final_label = final_labels[key]
            if isinstance(self.metric, _ThresholdScorer):
                if len(_final_label.shape) == 1:
                    _final_label = self.encoder.transform(np.reshape(_final_label, (len(_final_label), 1))).toarray()
            loss = []
            for i in range(n_base_model):
                if np.isnan(last_features[key][:, i*n_dim:(i+1)*n_dim]).any():
                    loss.append(-np.inf)
                else:
                    if self.task_type in CLS_TASKS:
                        if n_dim == 1:
                            tmp = last_features[key][:, i].reshape(-1, 1)
                            tmp = np.hstack([1-tmp, tmp])
                        else:
                            tmp = last_features[key][:, i*n_dim:(i+1)*n_dim]
                        if isinstance(self.metric, _PredictScorer):
                            tmp = np.argmax(tmp, axis=-1)
                    else:
                        tmp = last_features[key][:, i]
                    loss.append(self.metric._score_func(_final_label, tmp) * self.metric._sign)
            print(key, np.mean(loss), loss)

            losses[key] = loss

        return losses

    def register_leader(self, head_output, last_features, final_labels, layer):

        n_base_model = self.ensemble_size
        n_dim = last_features['train'].shape[1] // n_base_model

        # 计算val和test上的head输出
        head_outputs = {'train': head_output}
        for config in ['weighted', 'lightgbm', 'linear']:
            meta_learner = Blending.build_meta_learner(config, self.task_type, last_features['train'], final_labels['train'],
                                ensemble_size=n_base_model, if_imbal=self.if_imbal, metric=self.metric)

            for key in last_features.keys():
                pred_features = last_features[key]
                if config == 'weighted':
                    pred = meta_learner.stack_predict(pred_features)
                else:
                    if self.task_type in CLS_TASKS:
                        pred = meta_learner.predict_proba(pred_features)
                    else:
                        pred = meta_learner.predict(pred_features)

                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)

                head = f"{config}-L{layer}"
                if key not in head_outputs:
                    head_outputs[key] = {}
                head_outputs[key][head] = pred

        # best_perf = -np.inf
        best_config = None
        best_head = None
        best_last_feature = None
        # judge = 'train' if 'val' not in head_outputs else 'val'

        for key in head_outputs.keys():
            head_output = head_outputs[key]
            for head in head_output.keys():
                perf = self.cal_scores({key: head_output[head]}, {key: final_labels[key]}, n_base_model=1)[key][0]
                self.leader_board[key][head] = perf

        for head in head_outputs['train'].keys():
            meta_learner, _ = head.split('-')

            can_config = {'meta_learner': meta_learner, 'stack_layers': layer - 1, 
                          'train': self.leader_board['train'][head], 'val': self.leader_board['val'][head]}
            if better_ens(can_config, best_config):
                best_config = can_config
                best_head = head
                best_last_feature = last_features['train'].copy()

        if layer > 1:
            perfs = [(self.layer_loss[-1]['val'][_idx], self.layer_loss[-1]['train'][_idx]) for _idx in range(self.ensemble_size)]
            _best_idx = max(enumerate(perfs), key=lambda x:x[1])[0]

            head = f"best_idx{_best_idx}-L{layer}"

            for key in head_outputs.keys():
                perf = self.layer_loss[-1][key][_best_idx]
                self.leader_board[key][head] = perf

            can_config = {'meta_learner': 'best_idx', 'stack_layers': layer - 1, 
                        'train': self.layer_loss[-1]['train'][_best_idx], 'val': self.layer_loss[-1]['val'][_best_idx]}
            if better_ens(can_config, best_config):
                best_config = can_config
                best_head = head
                best_last_feature = np.full(last_features['train'].shape, np.nan)
                best_last_feature[:, _best_idx * n_dim:(_best_idx + 1) * n_dim] = last_features['train'][:, _best_idx * n_dim:(_best_idx + 1) * n_dim]

        return best_config, best_head, {'train': best_last_feature}

    def forward(self, base_features: dict, final_labels: dict=None, train=False, ori_xs: dict=None):

        _output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
        if train:
            assert 'train' in base_features
            assert final_labels is not None
            assert sorted(list(base_features.keys())) == sorted(list(final_labels.keys()))
            for key in base_features.keys():
                if key not in self.leader_board:
                    self.leader_board[key] = {}

        stack_configs = [[], ['weighted', 'lightgbm', 'linear']]
        model_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, _) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    stack_configs[0].append(config)
                model_cnt += 1
        n_base_model = len(stack_configs[0])
        _key = list(base_features.keys())[0]
        n_dim = base_features[_key].shape[1] // n_base_model
        last_features = deepcopy(base_features)
        new_features = {}

        best_config = None
        best_head = None
        best_last_features = last_features

        if train and not self.lock:
            self.layer_loss.append(self.cal_scores(base_features, final_labels, self.ensemble_size))
            self.stack_models = dict()

        if self.stack_layers > 0:
            for layer in range(self.stack_layers):

                if train:
                    new_node = DataManager(last_features['train'], final_labels['train']).get_data_node(last_features['train'], final_labels['train'])
                    if self.thread > 1: save_ori_x(ori_xs['train'], _output_dir)

                    fail_mask = np.full(n_base_model, False)
                    if self.lock:
                        sms = self.stack_models[f'layer_{layer+1}']
                        for i in range(len(sms)):
                            if sms[i] is None: fail_mask[i] = True

                    sms, _, new_feature, head_output, cost = layer_fit(stack_configs=stack_configs, new_node=new_node, ori_xs=ori_xs['train'], n_base_model=n_base_model,
                                                                    task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed,
                                                                    layer=layer+1, thread=self.thread, folds=self.sfolds, output_dir=_output_dir, logger=self.logger, metric=self.metric,
                                                                    skip_mask=fail_mask)
                    self.stack_models[f'layer_{layer+1}'] = sms

                    new_features = {'train': new_feature}
                    for key in last_features.keys():
                        if key != 'train':
                            new_features[key] = np.zeros_like(last_features[key])
                            for suc_cnt, config in enumerate(stack_configs[0]):
                                estimators = sms[suc_cnt]
                                if estimators is not None:
                                    for estimator in estimators:
                                        _new_node = new_node
                                        if ori_xs[key] is not None:
                                            _new_node = new_node.copy_(no_data=True)
                                            _new_node.data = (np.hstack([ori_xs[key][suc_cnt], last_features[key]]), final_labels[key])
                                        pred = fetch_predict_results(self.task_type, {}, estimator, _new_node)
                                        if len(pred.shape) == 1:
                                            pred = pred.reshape(-1, 1)
                                        new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)
                                else:
                                    new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = last_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim]

                    if not self.lock:
                        _best_config, _best_head, _best_last_features = self.register_leader(head_output, last_features, final_labels, layer+1)
                        if better_ens(_best_config, best_config):
                            best_config = _best_config
                            best_head = _best_head
                            best_last_features = _best_last_features
                        self.logger.info(f"Cost of Layer{layer+1} training with {self.thread} threads: {cost}s")
                        # print(head_outputs)

                        self.layer_loss.append(self.cal_scores(new_features, final_labels, self.ensemble_size))
                        self.train_cost.append(cost)

                        if self.retain:
                            judge = 'train' if 'val' not in new_features else 'val'
                            fail_mask = np.array(self.layer_loss[-1][judge]) < np.array(self.layer_loss[-2][judge]) + 1e-5
                            self.logger.info("Number of models getting improved: %d" % (n_base_model - fail_mask.sum()))
                        else:
                            fail_mask = np.full(n_base_model, False)
                        for t in range(n_base_model):
                            if fail_mask[t]: self.stack_models['layer_%d' % (layer+1)][t] = None

                    fail_mask = np.repeat(fail_mask, n_dim)
                    for key in new_features.keys():
                        last_features[key][:, ~fail_mask] = new_features[key][:, ~fail_mask]

                    if np.all(fail_mask):
                        self.logger.info("None model gets improved! early stop!")
                        for t in range(layer+1, self.stack_layers):
                            self.stack_models['layer_%d' % (t+1)] = [None] * n_base_model
                        break
                    elif layer == self.stack_layers - 1 and not self.lock:
                        new_node = DataManager(last_features['train'], final_labels['train']).get_data_node(last_features['train'], final_labels['train'])
                        _, _, _, head_output, cost = layer_fit(stack_configs=[[], stack_configs[1]], new_node=new_node, ori_xs=ori_xs['train'], n_base_model=n_base_model,
                                                            task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed,
                                                            layer=layer+2, thread=self.thread, folds=self.sfolds, output_dir=_output_dir, logger=self.logger, metric=self.metric)

                        _best_config, _best_head, _best_last_features = self.register_leader(head_output, last_features, final_labels, layer+2)
                        if better_ens(_best_config, best_config):
                            best_config = _best_config
                            best_head = _best_head
                            best_last_features = _best_last_features
                            # print(f"Head of layer{layer+2}: {head_outputs}")
                else:
                    new_node = DataManager(last_features['test'], None).get_data_node(last_features['test'], None)
                    sms = self.stack_models['layer_%d' % (layer+1)]
                    for key in last_features.keys():
                        new_features[key] = np.zeros_like(last_features[key])
                        for suc_cnt, config in enumerate(stack_configs[0]):
                            estimators = sms[suc_cnt]
                            if estimators is not None:
                                for estimator in estimators:
                                    _new_node = new_node
                                    if ori_xs is not None:
                                        _new_node = new_node.copy_(no_data=True)
                                        _new_node.data = (np.hstack([ori_xs[key][suc_cnt], last_features[key]]), None)
                                    pred = fetch_predict_results(self.task_type, {}, estimator, _new_node)
                                    if len(pred.shape) == 1:
                                        pred = pred.reshape(-1, 1)
                                    new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)
                            else:
                                new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = last_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim]
                    last_features = deepcopy(new_features)
                    best_last_features = last_features

        if train:
            if self.thread > 1:
                rm_ori_x(ori_xs['train'], _output_dir)

            if self.lock:
                best_last_features = last_features
            else:
                self.best_config = best_config
                tmp = best_head.split('-')
                if self.meta_method == 'auto':
                    self.meta_method = tmp[0]
                    best_layer = int(tmp[1][1:])
                    self.stack_layers = best_layer - 1
                    for key in self.stack_models.keys():
                        layer = int(key.split('_')[1])
                        if layer > self.stack_layers:
                            self.stack_models[key] = None

                if 'best' in self.meta_method:
                    best_idx = int(self.meta_method.split('_')[1][3:])
                    for i in range(len(self.stack_models[f'layer_{self.stack_layers}'])):
                        if i != best_idx:
                            self.stack_models[f'layer_{self.stack_layers}'][i] = None

                self.lock = True  # 锁定

        return best_last_features

    def fit(self, datanode, val_nodes: dict=None):
        if len(datanode.data[1].shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(datanode.data[1], (len(datanode.data[1]), 1))
            self.encoder.fit(reshape_y)
        # Train basic models using a part of training data
        base_features, ori_xs = self.get_base_features(datanode, val_nodes)

        final_labels = {'train': datanode.data[1]}
        if val_nodes is not None:
            for key in val_nodes.keys():
                final_labels[key] = val_nodes[key].data[1]
        last_features = self.forward(base_features, final_labels, train=True, ori_xs=ori_xs)

        self.meta_learner = self.build_meta_learner(self.meta_method, self.task_type, last_features['train'], final_labels['train'],
                                                    ensemble_size=self.ensemble_size, if_imbal=self.if_imbal, metric=self.metric)

        return self

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
                    _path = CombinedTopKModelSaver.get_parse_path(path, mode=mode, **self.resampling_params)
                    op_list, estimator, _ = CombinedTopKModelSaver._load(_path)
                    pred = fetch_predict_results(self.task_type, op_list, estimator, datanode)
                    if self.skip_connect:
                        if ori_xs is None: ori_xs = []
                        model_path = path if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(path, 'full')
                        op_list, _, _ = CombinedTopKModelSaver._load(model_path)
                        ori_x = construct_node(datanode.copy_(), op_list).data[0]
                        ori_xs.append(ori_x)
                    if len(pred.shape) == 1:
                        pred = pred.reshape(-1, 1)
                    n_dim = pred.shape[1] if pred.shape[1] > 2 else 1
                    if base_features is None:
                        num_samples = len(datanode.data[0])
                        base_features = np.zeros((num_samples, self.ensemble_size * n_dim))
                    base_features[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, -n_dim:]
                    suc_cnt += 1

                model_cnt += 1

        return base_features, ori_xs

    def predict(self, data, refit='full'):
        base_features, ori_xs = self.get_feature(data, refit)
        last_features = self.forward({'test': base_features}, ori_xs={'test': ori_xs})
        # Get predictions from meta-learner
        if self.meta_method == 'weighted' or self.meta_method.startswith('best_'):
            final_pred = self.meta_learner.stack_predict(last_features['test'])
        else:
            if self.task_type in CLS_TASKS:
                final_pred = self.meta_learner.predict_proba(last_features['test'])
            else:
                final_pred = self.meta_learner.predict(last_features['test'])
        return final_pred

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, model_path) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    ens_config.append((algo_id, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'blending'
        ens_info['stask_layers'] = self.stack_layers
        ens_info['meta_learner'] = self.meta_method
        judge = 'train' if 'val' not in self.leader_board else 'val'
        sorted_head = sorted(list(self.leader_board['train'].keys()), key=lambda x: (-self.leader_board[judge][x], -self.leader_board['train'][x]))
        ens_info['leader_board'] = [f"{head}: {', '.join(['%s-%.5f' % (key, self.leader_board[key][head]) for key in self.leader_board.keys()])}" for head in sorted_head]
        if self.ensemble_method == 'weighted':
            ens_info['meta_weighted'] = ','.join(['%.3f' % tmp for tmp in self.meta_learner.weights_])
        ens_info['thread'] = self.thread
        ens_info['train_cost'] = self.train_cost
        ens_info['layer_loss'] = self.layer_loss
        ens_info['config'] = ens_config
        return ens_info

