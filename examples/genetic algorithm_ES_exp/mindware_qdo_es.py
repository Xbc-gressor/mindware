'''
Important !!!!!
For this algorithm, there are one important thing to note,
For the lib ribs, you need change it as follow: 

    we need to revise the ribs.schedulers._scheduler, copy the bellow phase to replace the tell() in line 335.

    def tell(self, objective, measures, **fields):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective and measures arrays must be in the same order as
            the solutions created by :meth:`ask_dqd`; i.e. ``objective[i]`` and
            ``measures[i]`` should be the objective and measures for
            ``solution[i]``.

        Args:
            objective ((batch_size,) array): Each entry of this array contains
                the objective function evaluation of a solution.
            measures ((batch_size, measures_dm) array): Each row of this array
                contains a solution's coordinates in measure space.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
            ValueError: One of the inputs has the wrong shape.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        data = self._validate_tell_data({
            "objective": objective,
            "measures": measures,
            **fields,
        })

        # add_info = self._add_to_archives(data)

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self._emitters, self._num_emitted):
            end = pos + n
            emitter.tell(
                **{
                    name: arr[pos:end] for name, arr in data.items()
                },
                # add_info={
                #     name: arr[pos:end] for name, arr in add_info.items()
                # },
            )
            pos = end

'''

import sys
import os
sys.path.append('/home/new_mindware/mindware/')
from typing import Union, Callable

import numpy as np


from mindware.components.utils.constants import *
from mindware.utils.data_manager import DataManager
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.utils.topk_saver import CombinedTopKModelSaver, check_mode
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.feature_engineering.parse import construct_node
from mindware.modules.base_evaluator import fetch_predict_results
from mindware.modules.qdo_es.mindware_qdo_es import QDO_ES



if __name__ == '__main__':
    import pickle as pkl
    
    stats_path = '/home/test/new_data/CASHFE-block_0(1)-holdout_spambase_2025-06-06-18-42-53-435848/2025-06-06-18-42-53-435848_topk_config.pkl'
    with open(stats_path, 'rb') as f:
        stats = pkl.load(f)

    datasets_dir = '/home/kaggle_data/sub_automl_data/'

    dataset_path = os.path.join(datasets_dir, 'cls_datasets', 'spambase' + '.csv')
    dm = DataManager()
    task_type = CLASSIFICATION
    header = None
    df = dm.load_csv(dataset_path, header=header)

    train_df, test_df = dm.split_data(df, label_col=-1, test_size=0.2, random_state=42, task_type=task_type)

    train_data_node = dm.from_train_df(train_df, label_col=-1)
    train_data_node = dm.preprocess_fit(train_data_node, task_type)
        
       
    test_data_node = dm.from_test_df(test_df, has_label=True)
    test_data_node = dm.preprocess_transform(test_data_node)

    cma_es = QDO_ES(
        stats= stats, n_iterations= 10, batch_size=25, task_type=CLASSIFICATION, data_node = train_data_node
    )
    cma_es.get_weights()
    cma_es.refit(train_data_node, mode='full')
    p =cma_es.predict(test_data_node)
    print(p)
