from sklearn.metrics._scorer import _BaseScorer
import time
import datetime
import pandas as pd
from typing import List
from mindware.utils.logging_utils import get_logger
from mindware.components.dataset_discovery.dataset_discovery_builder import DatasetDiscoveryBuilder


class BaseFeaDiscovery(object):
    """Base class for feature selection."""

    def __init__(self, data_node,
                 data_source: List[pd.DataFrame],
                 fea_sel_method: str,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 random_state=1):
        self.data_node = data_node
        self.data_source = data_source
        self.fea_sel_method = fea_sel_method
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir
        self.random_state = random_state

        self.dataset_discovery = DatasetDiscoveryBuilder(data_node=self.data_node,
                                                         data_source=self.data_source,
                                                         dataset_sel_method='auto',
                                                         task_type=self.task_type,
                                                         metric=self.metric,
                                                         output_dir=self.output_dir)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        logger_name = 'FeaSelectionBuilder'
        self.logger = get_logger(logger_name)

    def discovery(self, data):
        raise NotImplementedError()
