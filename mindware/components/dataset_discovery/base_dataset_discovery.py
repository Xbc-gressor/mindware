from sklearn.metrics._scorer import _BaseScorer

import pandas as pd
from typing import List


class BaseDatasetDiscovery:
    def __init__(self, data_node,
                 data_source: List[pd.DataFrame],
                 dataset_sel_method: str,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        pass

    def discovery(self, data_node):
        return None

    def get_base_node_id(self):
        return None

    def get_table_by_id(self, id):
        return None

