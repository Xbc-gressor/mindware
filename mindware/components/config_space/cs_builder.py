from mindware.components.config_space.cls_cs_builder import get_cash_cs as get_cls_cash_cs
from mindware.components.config_space.rgs_cs_builder import get_cash_cs as get_rgs_cash_cs

from mindware.components.config_space.cls_cs_builder import get_fe_cs as get_cls_fe_cs
from mindware.components.config_space.rgs_cs_builder import get_fe_cs as get_rgs_fe_cs
from mindware.components.utils.constants import *

import numpy as np


def get_cash_cs(include_algorithms=None, task_type=CLASSIFICATION, **cs_args):

    resampling_params = cs_args.get('resampling_params', None)
    data_node = cs_args.get('data_node', None)

    test_size = 0.33
    if resampling_params is not None and 'test_size' in resampling_params:
        test_size = resampling_params['test_size']

    _cs_args = {}
    if data_node is not None:
        n_samples = int(data_node.data[0].shape[0] * (1 - test_size))
        _cs_args = {
            'n_classes': np.unique(data_node.data[1]).shape[0],
            'n_features': data_node.data[0].shape[1],
            'n_samples': n_samples
        }

    if task_type in CLS_TASKS:
        cs = get_cls_cash_cs(include_algorithms, task_type, **_cs_args)
    else:
        cs = get_rgs_cash_cs(include_algorithms, task_type, **_cs_args)

    return cs


def get_fe_cs(task_type=CLASSIFICATION, include_image=False,
              include_text=False, include_preprocessors=None, if_imbal=False):

    if task_type in CLS_TASKS:
        cs = get_cls_fe_cs(task_type, include_image=include_image, include_text=include_text,
                           include_preprocessors=include_preprocessors, if_imbal=if_imbal)
    else:
        cs = get_rgs_fe_cs(task_type, include_image=include_image, include_text=include_text,
                           include_preprocessors=include_preprocessors)

    return cs
