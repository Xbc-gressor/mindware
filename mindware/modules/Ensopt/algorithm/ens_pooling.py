from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from sklearn.metrics._scorer import _BaseScorer
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
import os

class avging(BaseEnsembleModel):
    def __init__(self, ens_size=10):
        self.model_pool = [None for _ in range(ens_size)]
        self.model_config = [None for _ in range(ens_size)]
        self.model_pred = [None for _ in range(ens_size)]

    def delete_model(self, model_id, output_dir = None, datetime=None):
        # if output_dir is not None and self.model_config[model_id] is not None:
        #     model_path = CombinedTopKModelSaver.get_path_by_config(output_dir, self.model_config[model_id], datetime, mode='partial')
        #     os.remove(model_path)
        #     print(f'Delete model {model_path}')
        self.model_pool[model_id] = None
        self.model_config[model_id] = None
        self.model_pred[model_id] = None
        self.model_id = model_id
        
    def replace_model(self, model, config, pred, model_id=None):
        if not model_id:
            self.model_pool[self.model_id] = model
            self.model_config[self.model_id] = config
            self.model_pred[self.model_id] = pred
        else:
            if model_id < len(self.model_pool):
                self.model_pool[model_id] = model
                self.model_config[model_id] = config
                self.model_pred[model_id] = pred
            else:
                raise ValueError(f"Model ID {model_id} is out of bounds for the current model pool.")
