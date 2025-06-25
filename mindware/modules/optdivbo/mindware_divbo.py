
from typing import List, Union, Callable
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
from mindware.components.metrics.metric import get_metric
from mindware.modules.optdivbo.utils.evaluate import evaluate
from mindware.modules.optdivbo.utils.data_loader import load_train_test_data
from functools import partial
import os
import numpy as np
from mindware.modules.optdivbo.ensembles.ensemble_selection import EnsembleSelection
from mindware.modules.base import BaseAutoML
from mindware.components.utils.topk_saver import load_combined_transformer_estimator, CombinedTopKModelSaver
from mindware.modules.base import fetch_predict_results



class Optdivbo(BaseAutoML):

    name='OptDivBO'
    def __init__(self, iter_num,  ens_size, include_algorithms,
                 task_type, data_node:DataNode, test_node: DataNode,
                 time_limit_per_trial,
                 alpha = 0.05, beta =0.2, 
                 metric:str = 'acc',  resampling_params=None,
                 task_name = 'default', time_limit=600, output_dir = './data', seed=1, task_id='test', include_preprocessors=None, filter_params=None
                 ):
        super(Optdivbo, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation='holdout', resampling_params=resampling_params,
            time_limit=time_limit,
            output_dir=output_dir, seed=seed,
            task_id=task_id
        )
        
        
        if task_type in CLS_TASKS:
            assert metric in ['cross_entropy', 'acc'], f'Optdivbo not support metric {metric}'

        if task_type in RGS_TASKS:
            assert metric in ['mse', 'mae'], f'Optdivbo not support metric {metric}'
        
        self.metric_func = get_metric(metric)

        self.task_type = task_type
        self.iter_num = iter_num
        self.time_limit_per_trial = time_limit_per_trial
        self.ens_size = ens_size

        path = 'OptDivBO-(%d)-%s_%s_%s' % (
            self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_fe_cs_args
        cs_args = get_fe_cs_args(**cs_args)
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
            include_algorithms = list(self.cs['algorithm'].choices)
        from mindware.components.config_space.cs_builder import get_fe_cs

        if self.filter_params is not None and 'n_preprocessor' in self.filter_params:
            n_prep = self.filter_params['n_preprocessor']
            include_preprocessors_dict = self._recommand_preps(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_prep=n_prep, include_algorithms=include_algorithms, include_preprocessors=include_preprocessors)
            include_preprocessors = []
            for preps in include_preprocessors_dict.values():
                include_preprocessors.extend(preps)
            include_preprocessors = list(set(include_preprocessors))
            
        tmp_cs = get_fe_cs(self.task_type, include_preprocessors=include_preprocessors, if_imbal=self.if_imbal, **cs_args)
        self.cs.add_hyperparameters(tmp_cs.get_hyperparameters())
        self.cs.add_conditions(tmp_cs.get_conditions())
        self.cs.add_forbidden_clauses(tmp_cs.get_forbiddens())

        self.test_node = test_node
        
        if task_type in CLS_TASKS:
            task_type_name = 'cls'
        else:
            task_type_name = 'rgs'
        self.task_type_name = task_type_name

        eval_func = partial(evaluate,
                            scorer=self.metric_func,
                            data_node=data_node, test_node=test_node, task_type=task_type_name,
                            resample_ratio=1.0, seed=1)

        _, _valid_node = self._get_train_valid_data(task_type_name, data_node)
        _y_val = _valid_node.data[1]
        self._y_val = _y_val


        from mindware.modules.optdivbo.divbo.bayesian_optimization_diversity import BayesianOptimizationDiversity
        self.optimizer = BayesianOptimizationDiversity(
                        config_space= self.cs,
                        eval_func= eval_func,
                        iter_num = self.iter_num,
                        score_name= metric,
                        task_name = task_name,
                        save_dir = self.output_dir,
                        surrogate_type = 'prf',
                        scorer = self.metric_func,
                        task_type= task_type_name,
                        ens_size= ens_size,
                        val_y_labels=_y_val,
                        alpha=alpha,
                        beta=beta
                    )
        
    def _get_train_valid_data(self, task_type,  data_node, seed=1):
        test_size = 0.33
        
        train_data = data_node.copy_(no_data=True)
        valid_data = data_node.copy_(no_data=True)
        X, y = data_node.data
        
        if task_type == 'cls':
            from sklearn.model_selection import StratifiedShuffleSplit
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                             random_state=seed)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=test_size,
                                   random_state=seed)

        for train_index, val_index in ss.split(X, y):
            train_data.data = [X[train_index], y[train_index]]
            valid_data.data = [X[val_index], y[val_index]]

        return train_data, valid_data

    def run(self):
        import pickle as pkl
        self.optimizer.run(time_limit_per_trial=self.time_limit_per_trial, total_time_limit = self.time_limit)
        save_path = self.optimizer.save_path
        with open(save_path, 'rb') as f:
            observations = pkl.load(f)

        with open(save_path, 'wb') as f:
            pkl.dump([observations, self._y_val, self.test_node.data[1]], f)

        with open(save_path, 'rb') as f:
            observations, val_labels, test_labels = pkl.load(f)
        config_list = []
        val_pred_list = []
        test_pred_list = []

        best_val = np.inf
        best_test = np.inf
        for ob in observations:
            config, val_perf, test_perf, val_pred, test_pred, _ = ob
            if val_pred is not None:
                config_list.append(config)
                val_pred_list.append(val_pred)
                test_pred_list.append(test_pred)
            if val_perf < best_val:
                best_val = val_perf
                best_test = test_perf

        self.ens = self.optimizer.ensemble
        self.ens.fit(val_pred_list, val_labels)

        for idx, config in enumerate(config_list):
            if idx not in self.ens.model_idx:
                continue
            if config is None:
                continue
            if not isinstance(config, dict):
                config = config.get_dictionary().copy()
            else:
                config = config.copy()
            algo_id = config['algorithm']
            if algo_id != 'neural_network':
                try:
                    op_list, estimator = self._refit_config(config, self.data_node, task_type=self.task_type,
                                                            if_imbal=self.if_imbal, resampling_params=self.resampling_params, seed=self.seed, mode='full')
                    pred = fetch_predict_results(self.task_type, op_list, estimator, self.test_node)
                    test_pred_list[idx] = pred
                except:
                    print("Error when refit incumbent config, use origin model!")

        ens_test_pred = self.ens.predict(test_pred_list)
        
        result = None
        if self.task_type in CLS_TASKS:
            result = np.argmax(ens_test_pred, axis=-1)
        else:
            result = ens_test_pred

        return self.test_node, result
