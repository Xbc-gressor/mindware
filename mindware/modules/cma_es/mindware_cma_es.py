import numpy as np

from mindware.components.utils.constants import *
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.utils.topk_saver import CombinedTopKModelSaver, check_mode
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.feature_engineering.parse import construct_node
from mindware.modules.base_evaluator import fetch_predict_results

from mindware.modules.cma_es.numerical_solvers.cmaes import CMAES
from mindware.modules.cma_es.util.metrics import make_metric

def get_metric(metric:str):
    if metric in ["accuracy", "acc"]:
        from sklearn.metrics import accuracy_score
        m = accuracy_score
        maximize = True
        cls = True
        always_transform_conf_to_pred = True
        optimum_value = 1

    elif metric == 'auc':
        from sklearn.metrics import roc_auc_score
        m = roc_auc_score
        maximize = True
        cls = True
        always_transform_conf_to_pred = False
        optimum_value = 1
    
    elif metric in ["mean_squared_error", "mse"]:
        from sklearn.metrics import mean_squared_error
        m = mean_squared_error
        maximize = False
        cls = False
        always_transform_conf_to_pred = False
        optimum_value = 0

    elif metric == 'r2':
        from sklearn.metrics import r2_score
        m = r2_score
        maximize = True
        always_transform_conf_to_pred = False
        cls = False
        optimum_value = 1

    return make_metric(m, metric, maximize, cls, always_transform_conf_to_pred,
                       optimum_value)

SUPPORT_METRIC = ['acc','accuracy','auc','mean_squared_error','mse','r2']
        



class CMA_ES(BaseEnsembleModel):
    
    def __init__(self, stats, n_iterations:int, batch_size:int, task_type, data_node:DataNode,
                 metric:str = 'acc', evaluation:str ='holdout',
                 resampling_params =dict(), 
                 output_dir=None,
                 if_imbal=False,
                 seed=1, n_jobs=1,):
        

        super().__init__(stats,
                         ensemble_method='CMA_ES',
                         ensemble_size=50,
                         task_type=task_type,
                         if_imbal= if_imbal,
                         metric=None,
                         resampling_params=resampling_params,
                         output_dir=output_dir,
                         seed=seed
                         )

        self.n_iterations = n_iterations
        self.batch_size =  batch_size
        
        assert metric in SUPPORT_METRIC,f"Not support this metric :{metric}"
        
        self.metric = get_metric(metric)
        
        self.data_node = data_node.copy_()
        self.n_jobs = n_jobs
    

        #TODO: only support holdout now.
        self.resampling_strategy = evaluation

        self.train_data = self.data_node.copy_(no_data=True)
        self.val_data = self.data_node.copy_(no_data=True)

        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        ss = self._get_spliter('holdout', test_size=test_size, random_state=self.seed)

        _x_train, _y_train = None, None
        _x_val, _y_val = None, None
        for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
            _x_train, _y_train = self.data_node.data[0][train_index], self.data_node.data[1][train_index]
            _x_val, _y_val = self.data_node.data[0][test_index], self.data_node.data[1][test_index]
        self.train_data.data = [_x_train, _y_train]
        self.val_data.data = [_x_val, _y_val]


        self.predictions, base_models = self.build_predictions(self.stats, self.val_data, self.task_type)

        self.model = CMAES(
            base_models, n_iterations = self.n_iterations,
            score_metric = self.metric,
            batch_size= self.batch_size,
            n_jobs = n_jobs,
            normalize_weights='softmax',
            trim_weights = 'ges-like-raw',
            random_state=np.random.RandomState(seed)
        )


    def _get_spliter(self, resampling_strategy, **kwargs):

        if self.task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)

        return ss
    
    def build_predictions(self, stats, valid_node, task_type):
        predictions = []
        base_models = []
        model_cnt = 0
        for algo_id in stats.keys():
            model_to_eval = stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                op_list, model, _ = CombinedTopKModelSaver._load(path)
                base_models.append(model)
                _node = valid_node.copy_()
                _node = construct_node(_node, op_list)
                X_valid, y_valid = _node.data

                if task_type in CLS_TASKS:
                    y_valid_pred = model.predict_proba(X_valid)
                else:
                    y_valid_pred = model.predict(X_valid)
                predictions.append(y_valid_pred)

                model_cnt += 1

        return predictions, base_models
    

    def get_weights(self):
        y_val = self.val_data.data[1]
        self.model.ensemble_fit(self.predictions, y_val)
        self.weights_ = self.model.weights_
        self.base_model_mask = np.full(len(self.predictions), False)
        self.base_model_mask[self.weights_ != 0] = True
        return self.model.weights_
    
    def predict(self, data, refit='full'):
        predictions = []
        cur_idx = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[cur_idx]:
                    path = CombinedTopKModelSaver.get_parse_path(path, mode=refit, **self.resampling_params)
                    op_list, estimator, _ = CombinedTopKModelSaver._load(path)
                    predictions.append(fetch_predict_results(self.task_type, op_list, estimator, data))
                cur_idx += 1
        predictions = np.asarray(predictions)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
