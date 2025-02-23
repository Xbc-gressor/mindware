import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch import RankNetAdvisor
from mindware import REGRESSION, CLASSIFICATION

task = 'cls'

task_type = CLASSIFICATION
if task == 'rgs':
    task_type = REGRESSION

chosen_datasets = ['kc1', 'cpu_act', 'ailerons', 'sick', 'mv', 'covertype']  # , 'higgs', 'spambase'
if task == 'rgs':
    chosen_datasets = ['debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'Moneyball', 'black_friday']

alad = RankNetAdvisor(task_type=CLASSIFICATION, n_algorithm=5, metric='acc', exclude_datasets=chosen_datasets)
alad.fit()
breakpoint()
# model_candidates = alad.fetch_algorithm_set(dataset_id, datanode=train_data)