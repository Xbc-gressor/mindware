from sklearn.metrics._scorer import _BaseScorer

from mindware.components.feature_discovery.arda import ARDA
import pandas as pd
from typing import List

ensemble_list = ['arda']


class FeaDiscoveryBuilder:
    def __init__(self, data_node,
                 data_source: List[pd.DataFrame],
                 fea_dis_method: str,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        self.discovery = None
        if fea_dis_method == 'arda':
            self.discovery = ARDA(data_node=data_node,
                                 data_source=data_source,
                                 fea_sel_method=fea_dis_method,
                                 task_type=task_type,
                                 metric=metric,
                                 output_dir=output_dir)
        else:
            raise ValueError("%s is not supported for feature selection!" % fea_dis_method)

    def discovery(self, data):
        return self.discovery.discovery(data)
