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
from mindware.modules.cma_es.mindware_cma_es import CMA_ES



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

    cma_es = CMA_ES(
        stats= stats, n_iterations= 10, batch_size=25, task_type=CLASSIFICATION, data_node = train_data_node
    )

    cma_es.get_weights()
    cma_es.refit(train_data_node, mode='full')
    p =cma_es.predict(test_data_node)
    print(p)
