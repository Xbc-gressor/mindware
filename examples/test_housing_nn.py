import pandas as pd
import os
import sys
import time

# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.utils.data_manager import DataManager
from mindware import REGRESSION
from mindware.components.metrics.metric import get_metric
from mindware.components.models.regression.neural_network import NeuralNetworkClassifier
from mindware.modules.hpo.hpo_evaluator import HPORGSEvaluator
from mindware.components.config_space.cs_builder import get_hpo_cs


x_encode = 'normalize'
label_encode = 'normalize'
ensemble_method = "ensemble_selection"
ensemble_size = 5
metric = 'rmse'
evaluation = 'holdout'
estimator_id = 'neural_network'

task_type = REGRESSION

data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/houseprice'

dm = DataManager()

ori_train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['Id'],
                                    label_name='SalePrice')
train_data_node = dm.preprocess_fit(ori_train_data_node, task_type, x_encode=x_encode, label_encode=label_encode)

ori_test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['Id'])
test_data_node = dm.preprocess_transform(ori_test_data_node)

evaluator = HPORGSEvaluator(
                fixed_config=None,
                scorer=get_metric(metric),
                data_node=train_data_node,
                resampling_strategy=evaluation,
                resampling_params={'test_size': 0.25},
                timestamp=time.time(),
                output_dir='./data')

cs = get_hpo_cs(estimator_id, task_type)

default = cs.get_default_configuration()
print(evaluator(default))

breakpoint()