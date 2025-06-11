from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from sklearn.metrics._scorer import _BaseScorer
from mindware.components.utils.constants import CLS_TASKS
class avging(BaseEnsembleModel):
    def __init__(self, ens_size=10):
        self.model_pool = [None for _ in range(ens_size)]

    def delete_model(self, model_id):
        self.model_pool[model_id] = None
        self.model_id = model_id
        
    def replace_model(self, model, model_id=None):
        if not model_id:
            self.model_pool[self.model_id] = model
        else:
            if model_id < len(self.model_pool):
                self.model_pool[model_id] = model
            else:
                raise ValueError(f"Model ID {model_id} is out of bounds for the current model pool.")
