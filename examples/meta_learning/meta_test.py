import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch import RankNetAdvisor as RankNetAdvisor_CV
from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch_full import RankNetAdvisor as RankNetAdvisor_FULL
from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch_weight import RankNetAdvisor as RankNetAdvisor_W
from mindware.components.meta_learning.algorithm_recomendation.gbm_advisor import GBMAdvisor
from mindware import REGRESSION, CLASSIFICATION

task = 'rgs'
metric = 'mse'
topk = 6

fix_tops = [
    'adaboost', 'extra_trees', 'gradient_boosting',
    'random_forest', 'lightgbm', 'xgboost'
]
task_type = CLASSIFICATION
if task == 'rgs':
    task_type = REGRESSION

chosen_datasets = ['kc1', 'cpu_act', 'ailerons', 'sick', 'mv', 'covertype']  # , 'higgs', 'spambase'
if task == 'rgs':
    chosen_datasets = ['debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'Moneyball', 'black_friday']

save_path1 = os.path.join('/root/xbc/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec', '%s_meta_dataset_embedding.pkl' % task)
with open(save_path1, 'rb') as f:
    data1 = pkl.load(f)
task_ids = [t[5:] for t in data1['task_ids']]
# 读取 Excel 文件中的特定 sheet
datasets_info = pd.read_excel('/root/xbc/数据集.xlsx', sheet_name='CLS' if task=='cls' else 'REG').set_index('Datasets')
datasets_info = datasets_info.loc[task_ids].sort_values(by='Instances')


mine_counts = []
mine_rks = []
mine_top1 = 0

minef_counts = []
minef_rks = []
minef_top1 = 0

minew_counts = []
minew_rks = []
minew_top1 = 0

mineg_counts = []
mineg_rks = []
mineg_top1 = 0

fix_counts = []
fix_rks = []
fix_top1 = 0

data_count = 0
for start in [0, 1, 2, 3, 4]:
    chosen_datasets = datasets_info.index[range(start, len(datasets_info.index), 5)]
    data_count += len(chosen_datasets)

    alad_full = RankNetAdvisor_FULL(task_type=task_type, n_algorithm=5, metric=metric, exclude_datasets=chosen_datasets)
    alad_full.fit()
    alad = RankNetAdvisor_CV(task_type=task_type, n_algorithm=5, metric=metric, exclude_datasets=chosen_datasets)
    alad.fit()
    alad_w = RankNetAdvisor_W(task_type=task_type, n_algorithm=5, metric=metric, exclude_datasets=chosen_datasets)
    alad_w.fit()
    alad_g = GBMAdvisor(task_type=task_type, n_algorithm=5, metric=metric, exclude_datasets=chosen_datasets)
    alad_g.fit()

    for cd in chosen_datasets:
        model_candidates = alad.fetch_algorithm_set(cd, datanode=None)
        model_candidatesf = alad_full.fetch_algorithm_set(cd, datanode=None)
        model_candidatesw = alad_w.fetch_algorithm_set(cd, datanode=None)
        model_candidatesg = alad_g.fetch_algorithm_set(cd, datanode=None)
        real_ranks = list(alad_w.fetch_run_results(cd).keys())

        # 计算top6找出多少个
        real_tops = real_ranks[:topk]
        mine_tops = model_candidates[:topk]
        minef_tops = model_candidatesf[:topk]
        minew_tops = model_candidatesw[:topk]
        mineg_tops = model_candidatesg[:topk]

        mine_count = 0
        minef_count = 0
        minew_count = 0
        mineg_count = 0
        fix_count = 0
        for t in real_tops:
            if t in mine_tops:
                mine_count += 1
            if t in minef_tops:
                minef_count += 1
            if t in minew_tops:
                minew_count += 1
            if t in mineg_tops:
                mineg_count += 1
            if t in fix_tops:
                fix_count += 1

        mine_counts.append(mine_count)
        minef_counts.append(minef_count)
        mineg_counts.append(mineg_count)
        fix_counts.append(fix_count)
        minew_counts.append(minew_count)

        mine_rk = [real_ranks.index(t) + 1 for t in mine_tops]
        minef_rk = [real_ranks.index(t) + 1 for t in minef_tops]
        minew_rk = [real_ranks.index(t) + 1 for t in minew_tops]
        mineg_rk = [real_ranks.index(t) + 1 for t in mineg_tops]
        fix_rk = [real_ranks.index(t) + 1 for t in fix_tops]
        mine_rks.append(np.sum(mine_rk))
        minef_rks.append(np.sum(minef_rk))
        minew_rks.append(np.sum(minew_rk))
        mineg_rks.append(np.sum(mineg_rk))
        fix_rks.append(np.sum(fix_rk))

        mine_top1 += real_tops[0] in mine_tops
        minef_top1 += real_tops[0] in minef_tops
        minew_top1 += real_tops[0] in minew_tops
        mineg_top1 += real_tops[0] in mineg_tops
        fix_top1 += real_tops[0] in fix_tops

        # print(cd, model_candidates)
        # print(" " * len(cd), real_ranks)

print("MetaW top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(minew_counts), np.mean(minew_rks), minew_top1/data_count))
print("Meta  top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(mine_counts), np.mean(mine_rks), mine_top1/data_count))
print("MetaG top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(mineg_counts), np.mean(mineg_rks), mineg_top1/data_count))
print("MetaF top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(minef_counts), np.mean(minef_rks), minef_top1/data_count))
print("Fix   top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(fix_counts), np.mean(fix_rks), fix_top1/data_count))

"""
CLS:

chosen_datasets = ['kc1', 'cpu_act', 'ailerons', 'sick', 'mv', 'covertype']

Meta top6 count:5.00; rank:27.83
Fix  top6 count:5.00; rank:25.50

range(2, len, 5)
Meta top6 count:4.05; rank:30.55
Fix  top6 count:4.00; rank:31.73
Meta top6 count:3.91 | rank:31.32 | top1 recall:0.77  # 早停的checkpoint甚至不如上面训练到最后的
Fix  top6 count:4.00 | rank:31.73 | top1 recall:0.64

但是对比用所有数据不分验证集，效果又不好
Meta  top6 count:3.95 | rank:31.27 | top1 recall:0.77  patience 10
Meta  top6 count:4.00 | rank:30.77 | top1 recall:0.82  patience 30
MetaW top6 count:4.09 | rank:30.14 | top1 recall:0.82  patience 30+weight0.5-1.5
MetaW top6 count:4.14 | rank:29.77 | top1 recall:0.77  patience 30+weight0-2
MetaW top6 count:5.59 | rank:23.18 | top1 recall:1.00  patience 30+weight0.5-1.5 用所有数据
MetaF top6 count:3.82 | rank:32.59 | top1 recall:0.68
Fix   top6 count:4.00 | rank:31.73 | top1 recall:0.64

新数据
Meta  top6 count:4.00 | rank:31.14 | top1 recall:0.77
MetaF top6 count:3.91 | rank:31.59 | top1 recall:0.77
MetaW top6 count:4.09 | rank:29.59 | top1 recall:0.86
Fix   top6 count:4.00 | rank:31.73 | top1 recall:0.64

新数据上全数据集cv测试
MetaW top6 count:4.00 | rank:30.95 | top1 recall:0.78
MetaG top6 count:4.06 | rank:31.01 | top1 recall:0.69
Meta  top6 count:3.91 | rank:32.06 | top1 recall:0.69
MetaF top6 count:3.82 | rank:32.66 | top1 recall:0.64
Fix   top6 count:3.96 | rank:32.34 | top1 recall:0.63

RGS:

['debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'Moneyball', 'black_friday']

Meta top6 count:3.67; rank:32.83
Fix  top6 count:4.17; rank:29.83

range(2, len, 5)
MetaW top6 count:4.00 | rank:30.46 | top1 recall:0.92
MetaW top6 count:3.92 | rank:31.23 | top1 recall:0.77
Meta  top6 count:4.15 | rank:30.54 | top1 recall:0.77  patience 30
MetaG top6 count:4.00 | rank:30.69 | top1 recall:0.69
MetaF top6 count:3.92 | rank:31.54 | top1 recall:0.54
Fix   top6 count:4.23 | rank:30.62 | top1 recall:0.77

range(1, len, 5)
MetaW top6 count:3.62 | rank:33.77 | top1 recall:0.46
Meta  top6 count:3.77 | rank:32.54 | top1 recall:0.69
MetaF top6 count:3.85 | rank:31.92 | top1 recall:0.69
Fix   top6 count:4.08 | rank:30.15 | top1 recall:0.77

range(0, len, 5)
MetaW top6 count:3.93 | rank:31.93 | top1 recall:0.64
Meta  top6 count:4.07 | rank:30.64 | top1 recall:0.71
MetaF top6 count:4.07 | rank:30.71 | top1 recall:0.79
Fix   top6 count:4.21 | rank:30.57 | top1 recall:0.71

range(3, len, 5)
MetaW top6 count:3.77 | rank:32.38 | top1 recall:0.54
Meta  top6 count:3.77 | rank:31.77 | top1 recall:0.62
MetaF top6 count:4.08 | rank:31.00 | top1 recall:0.62
Fix   top6 count:4.69 | rank:27.08 | top1 recall:0.77

range(4, len, 5)
MetaW top6 count:3.62 | rank:34.15 | top1 recall:0.77
Meta  top6 count:3.54 | rank:35.08 | top1 recall:0.69
MetaF top6 count:3.62 | rank:34.46 | top1 recall:0.69
Fix   top6 count:3.85 | rank:32.38 | top1 recall:0.62

全数据集cv测试
MetaW top6 count:3.80 | rank:32.52 | top1 recall:0.61
Meta  top6 count:3.86 | rank:32.09 | top1 recall:0.70
MetaG top6 count:4.05 | rank:30.45 | top1 recall:0.70
MetaF top6 count:3.91 | rank:31.91 | top1 recall:0.67
Fix   top6 count:4.21 | rank:30.17 | top1 recall:0.73
"""