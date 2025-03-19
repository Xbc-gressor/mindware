import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.evaluators.base_evaluator import fetch_predict_estimator
from mindware.components.feature_engineering.parse import construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator


class Blending(BaseEnsembleModel):
    def __init__(self, stats, data_node,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None, seed=None,
                 resampling_params=None, ratio = 0.5,
                 meta_learner='lightgbm'):
        super().__init__(stats=stats,
                         data_node=data_node,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         resampling_params=resampling_params,
                         output_dir=output_dir, seed=seed, ratio = ratio)
        try:
            from lightgbm import LGBMClassifier
        except:
            warnings.warn("Lightgbm is not imported! Blending will use linear model instead!")
            meta_learner = 'linear'
        self.meta_method = meta_learner
        # We use Xgboost as default meta-learner
        if self.task_type in CLS_TASKS:
            if meta_learner == 'linear':
                try:
                    from sklearn.linear_model import LogisticRegression
                except:
                    from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                try:
                    from sklearn.ensemble import GradientBoostingClassifier
                except:
                    from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMClassifier
                self.meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150, n_jobs=1, verbose=-1)
        else:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMRegressor
                self.meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70, n_jobs=1)

    def get_path(self, algo_id, model_cnt):

        if algo_id in ['extra_trees']:
            _path = os.path.join(self.output_dir, '%s-blending-model%d.joblib' % (self.timestamp, model_cnt))
        else:
            _path = os.path.join(self.output_dir, '%s-blending-model%d.pkl' % (self.timestamp, model_cnt))

        return _path

    def fit(self, data):
        # Split training data for phase 1 and phase 2
        test_size = 0.2

        # Train basic models using a part of training data
        model_cnt = 0
        suc_cnt = 0
        feature_p2 = None
        y_p2 = None
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):

                if self.base_model_mask[model_cnt] == 1:
                    op_list, model, _ = CombinedTopKModelSaver._load(path)

                    _node = data.copy_()

                    _node = construct_node(_node, op_list, mode='train')

                    X, y = _node.data
                    
                    if self.task_type in CLS_TASKS:
                        ss = BaseCLSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=self.seed)
                    else:
                        ss = BaseRGSEvaluator._get_spliter(resampling_strategy='holdout', test_size=test_size, random_state=self.seed)

                    x_p1, x_p2, y_p1, y_p2 = None, None, None, None
                    for train_index, val_index in ss.split(X, y):
                        x_p1, y_p1 = X[train_index], y[train_index]
                        x_p2, y_p2 = X[val_index], y[val_index]

                    estimator = fetch_predict_estimator(self.task_type, algo_id, config, x_p1, y_p1,
                                                        weight_balance=_node.enable_balance,
                                                        data_balance=_node.data_balance)
                    _path = self.get_path(algo_id, model_cnt)
                    CombinedTopKModelSaver._save(items=estimator, save_path=_path)

                    if self.task_type in CLS_TASKS:
                        pred = estimator.predict_proba(x_p2)
                        n_dim = np.array(pred).shape[1]
                        if n_dim == 2:
                            # Binary classificaion
                            n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(x_p2)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        if n_dim == 1:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, 1:2]
                        else:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    else:
                        pred = estimator.predict(x_p2).reshape(-1, 1)
                        n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(x_p2)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    suc_cnt += 1

                model_cnt += 1

        self.meta_learner.fit(feature_p2, y_p2)

        return self

    def get_feature(self, data):
        # Predict the labels via blending
        feature_p2 = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    op_list, model, _ = CombinedTopKModelSaver._load(path)
                    _node = data.copy_()

                    _node = construct_node(_node, op_list)
                    _path = self.get_path(algo_id, model_cnt)
                    estimator = CombinedTopKModelSaver._load(_path)
                    if self.task_type in CLS_TASKS:
                        pred = estimator.predict_proba(_node.data[0])
                        n_dim = np.array(pred).shape[1]
                        if n_dim == 2:
                            # Binary classificaion
                            n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(data.data[0])
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        if n_dim == 1:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, 1:2]
                        else:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    else:
                        pred = estimator.predict(_node.data[0]).reshape(-1, 1)
                        n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(data.data[0])
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    suc_cnt += 1

                model_cnt += 1

        return feature_p2

    def predict(self, data):
        feature_p2 = self.get_feature(data)
        # Get predictions from meta-learner
        if self.task_type in CLS_TASKS:
            final_pred = self.meta_learner.predict_proba(feature_p2)
        else:
            final_pred = self.meta_learner.predict(feature_p2)
        return final_pred

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, _) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    model_path = self.get_path(algo_id, model_cnt)
                    ens_config.append((algo_id, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'blending'
        ens_info['config'] = ens_config
        ens_info['meta_learner'] = self.meta_method
        return ens_info

    def refit(self):
        self.logger.debug("Start to refit all models needed by ensemble, no need with blending!")
