from mindware.components.utils.constants import *

from mindware.components.feature_engineering.parse import parse_config, construct_node

from mindware.modules.base_evaluator import BaseCLSEvaluator
from mindware.modules.base_evaluator import BaseRGSEvaluator


class CASHFECLSEvaluator(BaseCLSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, reshuffle=False
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal, reshuffle=reshuffle
        )

    def _get_parse_data_node(self, config, record=True):
        data_node, op_list = parse_config(self.train_node, config, record=record, if_imbal=self.if_imbal)
        _val_node = self.val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node


class CASHFERGSEvaluator(BaseRGSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=REGRESSION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed
        )

    def _get_parse_data_node(self, config, record=True):
        data_node, op_list = parse_config(self.train_node, config, record=record)
        _val_node = self.val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node

