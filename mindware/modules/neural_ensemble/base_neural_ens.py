from mindware.components.utils.constants import CLS_TASKS
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from typing import Dict
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.modules.base_evaluator import fetch_predict_results
from mindware.components.metrics.metric import get_metric
from mindware.modules.neural_ensemble.regularized_neural_ensemblers.model import NeuralEnsembler
from mindware.modules.neural_ensemble.regularized_neural_ensemblers.trainer import Trainer
from mindware.modules.neural_ensemble.regularized_neural_ensemblers.trainer_args import TrainerArgs
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
import numpy as np
import torch
import json
import os
import time
import pickle as pkl
import concurrent.futures as cfutures
from mindware.components.ensemble.parallel_fit import parallel_predict


class Neural_ensemble(BaseEnsembleModel):
    # 实际上两种模式都是基于blending的结构做的
    name = 'Neural_ensemble'  
    def __init__(self, stats: Dict, data_node, mode: str = 'model_averaging', 
                 task_type: str = 'classification', metric: str = 'acc',
                 dropout_rate: float = 0.5, num_layers: int = 3, output_dir: str = None, if_imbal=False,
                 hidden_dim: int = 32, batch_size: int = 256, epoch: int = 1000, val_size: float = 0.25, seed: int = 1, n_jobs=1,
                 task_id='test'):
        
        super().__init__(stats,
                         ensemble_method='QDO_ES',
                         ensemble_size=50,
                         task_type=task_type,
                         if_imbal= if_imbal,
                         metric=None,
                         output_dir=output_dir,
                         seed=seed
                         )
        
        self.stats = stats
        self.mode = mode
        self.task_type = task_type
        self.seed = seed
        self.task_type = task_type
        self.data_node = data_node
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        

        self.val_size = val_size
        self.batch_size = batch_size

        self.thread = n_jobs
        self.task_id = task_id

        path = 'NEURAL_ES-b%d(%d)-%s_%s_%s' % (
            self.batch_size, self.seed, 'holdout', self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.cost = 0
        if isinstance(metric, str):
            self.metric_name = metric
        self.metric = get_metric(metric)
        # data解析
        val_data = self.data_node.copy_(no_data=True)
        ss = self._get_spliter('holdout', test_size=val_size, random_state=self.seed)
        for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
            _x_val, _y_val = self.data_node.data[0][test_index], self.data_node.data[1][test_index]
            val_data.data = [_x_val, _y_val]
        self.val_data_node = val_data
        
    def _get_spliter(self, resampling_strategy, **kwargs):

        if self.task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)

        return ss
    
    
    def build_predictions(self, stats, valid_node, task_type):
        start = time.time()

        predictions = []
        if self.thread == 1:
            model_cnt = 0
            for algo_id in stats.keys():
                model_to_eval = stats[algo_id]
                for idx, (config, _, path) in enumerate(model_to_eval):
                    op_list, model, _ = CombinedTopKModelSaver._load(path)
                    predictions.append(fetch_predict_results(task_type, op_list, model, valid_node))

                    model_cnt += 1
        else:
            output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            all_model_cnt = 0
            for algo_id in stats.keys():
                all_model_cnt += len(stats[algo_id])

            predictions = [None] * all_model_cnt
            node_path = os.path.join(output_dir, 'valid_node.pkl')
            with open(node_path, 'wb') as f:
                pkl.dump(valid_node, f)

            with cfutures.ProcessPoolExecutor(max_workers=self.thread) as executor:
                fs_wait = set()

                model_cnt = 0
                for algo_id in stats.keys():
                    model_to_eval = stats[algo_id]
                    for idx, (config, _, path) in enumerate(model_to_eval):
                        kwargs = {
                            'model_idx': model_cnt,'config_path': path, 'task_type': task_type, 'node_path': node_path
                        }

                        if len(fs_wait) < self.thread:
                            fs_wait.add(executor.submit(parallel_predict, **kwargs))
                        else:
                            fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                            fs_wait.add(executor.submit(parallel_predict, **kwargs))
                            for fs in fs_done:
                                model_idx, pred = fs.result()
                                predictions[model_idx] = pred

                        model_cnt += 1

                while len(fs_wait) > 0:
                    fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                    for fs in fs_done:
                        model_idx, pred = fs.result()
                        predictions[model_idx] = pred

            os.remove(node_path)

        print(f"Build predictions with {self.thread} threads cost: {time.time() - start}s!")

        return predictions
    
    def fit(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_pred_list = self.build_predictions(self.stats, self.val_data_node, self.task_type)

        if self.task_type in CLS_TASKS:
            model_pred_list = np.moveaxis(np.array(model_pred_list), 0, -1)
        else:
            model_pred_list = np.array(model_pred_list)[..., np.newaxis]
            model_pred_list = np.moveaxis(model_pred_list, 0, -1)
        num_samples, num_classes, num_base_functions = model_pred_list.shape
        start = time.time()
        model = NeuralEnsembler(num_base_functions=num_base_functions,
                                num_classes=num_classes,
                                hidden_dim=self.hidden_dim,
                                num_layers=self.num_layers,
                                dropout_rate=self.dropout_rate,
                                task_type=self.task_type, 
                                mode=self.mode).to(device)
        trainer_args = TrainerArgs(batch_size=self.batch_size, lr=0.001, epochs=self.epoch, device=device, task_id=self.task_id, task_type=self.task_type)
        trainer = Trainer(model=model, trainer_args=trainer_args)
        loss_lst = trainer.fit(model_pred_list, self.val_data_node.data[1])
        self.cost = time.time() - start
        self.model = model

    def predict(self, test_data):
        # 需要base model先在data上（test）预测得到base_functions_test
        base_functions_test = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                op_list, model, _ = CombinedTopKModelSaver._load(path)
                base_functions_test.append(fetch_predict_results(self.task_type, op_list, model, test_data))

        if self.task_type in CLS_TASKS:
            base_functions_test = np.moveaxis(np.array(base_functions_test), 0, -1)
        else:
            base_functions_test = np.array(base_functions_test)[..., np.newaxis]
            base_functions_test = np.moveaxis(base_functions_test, 0, -1)
        preds = self.model.predict(base_functions_test) 
        results = []
        for pred in preds:
            if self.task_type in CLS_TASKS:
                results.append(np.argmax(pred, axis=-1))
            else:
                results.append(pred)

        return results
    
    def run(self):
        self.fit()

    def get_conf(self, save=False):
        # 获取对象的配置信息
        conf = {
            'name': self.name,
            'task_type': self.task_type,
            'task_id': self.task_id,
            'metric': self.metric_name,
            'seed': self.seed,
            'dropout_rate': self.dropout_rate,
            'num_layers': self.num_layers,
            'epoch': self.epoch,
            'batch_size': self.batch_size,
            'hidden_dim': self.hidden_dim,
            'output_dir': self.output_dir,
            'val_size': self.val_size,

        }

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf
    def get_model_info(self, save=True):
        model_info = {'cost': self.cost}

        if save:
            with open(os.path.join(self.output_dir, 'best_model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=4)
        