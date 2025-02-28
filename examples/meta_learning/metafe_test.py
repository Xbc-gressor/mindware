import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import random
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mindware.components.meta_learning.fe_recomendation.gbm_advisor import GBMAdvisor
from mindware.components.meta_learning.fe_recomendation.base_advisor import _cls_builtin_algorithms, _rgs_builtin_algorithms
from mindware import REGRESSION, CLASSIFICATION
from mindware.components.config_space.cs_builder import get_fe_cs

task = 'cls'
metric = 'acc'
topk_r = 1/3
topk_pre = 6

random_tops = None

task_type = CLASSIFICATION
if task == 'rgs':
    task_type = REGRESSION

chosen_datasets = ['kc1', 'cpu_act', 'ailerons', 'sick', 'mv', 'covertype']  # , 'higgs', 'spambase'
if task == 'rgs':
    chosen_datasets = ['debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'Moneyball', 'black_friday']

builtin_algorithms = _cls_builtin_algorithms
if task == 'rgs':
    builtin_algorithms = _rgs_builtin_algorithms

save_path1 = os.path.join('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec', '%s_meta_dataset_embedding.pkl' % task)
with open(save_path1, 'rb') as f:
    data1 = pkl.load(f)
task_ids = [t[5:] for t in data1['task_ids']]
# 读取 Excel 文件中的特定 sheet
datasets_info = pd.read_excel('/root/automl_data/automl_data/数据集.xlsx', sheet_name='CLS' if task=='cls' else 'REG').set_index('Datasets')
datasets_info = datasets_info.loc[task_ids].sort_values(by='Instances')

save_path3 = os.path.join('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec', '%s_meta_dataset_preprocessor.pkl' % task)
with open(save_path3, 'rb') as f:
    data3 = pkl.load(f)

full_cs = get_fe_cs(task_type, include_preprocessors=None, meta = True, silence = True)
full_pres = [ x for x in list(full_cs['preprocessor'].choices) if x != 'empty' ]

mineg_counts = []
mineg_rks = []
mineg_top1 = 0
mineg_csnum = []
mineg_best = 0

random_counts = []
random_rks = []
random_top1 = 0
random_csnum = []
random_best = 0

fix_csnum = []
fix_best = 0

empty_num = 0

data_count = 0
sims = []
for algo in builtin_algorithms:
    for start in range(10):
        chosen_datasets = datasets_info.index[range(start, len(datasets_info.index), 10)]
        train_datasets = [t for t in datasets_info.index if t not in chosen_datasets]
        topk = int(len(train_datasets) * topk_r)

        data_count += len(chosen_datasets)

        alad_g = GBMAdvisor(task_type=task_type, metric=metric, exclude_datasets=chosen_datasets, include_algorithms=[algo])  
        alad_g.fit()

        for cd in chosen_datasets:
            sel_idx1 = data3['task_ids'].index('init_'+cd)
            sel_idx2 = data3['algorithms_included'].index(algo)
            best = data3['best_preprocessor'][metric][sel_idx1][sel_idx2]
            if best == 'empty':
                empty_num += 1

            random_tops = random.sample(train_datasets, topk)
            random_toppres = random.sample(full_pres, topk_pre) + ['empty']
            fix_toppres = alad_g.metadata_manager._sup_preprocessor[algo][:topk_pre] + ['empty']
            precprocessors = alad_g.fetch_preprocessor_set(cd)[algo][:topk_pre] + ['empty']
            print(sorted(precprocessors))

            predict = alad_g.fetch_dataset_set(cd, datanode=None)[algo]
            model_candidatesg = list(predict.keys())
            sims.extend(list(predict.values()))

            if best in precprocessors:
                mineg_best += 1
            mineg_cs = get_fe_cs(task_type, include_preprocessors=precprocessors, meta = True, silence = True)
            mineg_csnum.append(len(mineg_cs))

            if best in random_toppres:
                random_best += 1
            random_cs = get_fe_cs(task_type, include_preprocessors=random_toppres, meta = True, silence = True)
            random_csnum.append(len(random_cs))

            if best in fix_toppres:
                fix_best += 1
            fix_cs = get_fe_cs(task_type, include_preprocessors=fix_toppres, meta = True, silence = True)
            fix_csnum.append(len(fix_cs))

            real_ranks = list(alad_g.fetch_run_results(cd)[algo].keys())
            # 计算top6找出多少个
            real_tops = real_ranks[:topk]
            mineg_tops = model_candidatesg[:topk]

            mineg_count = 0
            random_count = 0
            for t in real_tops:
                if t in mineg_tops:
                    mineg_count += 1
                if t in random_tops:
                    random_count += 1

            mineg_counts.append(mineg_count)
            random_counts.append(random_count)

            mineg_rk = [real_ranks.index(t) + 1 for t in mineg_tops]
            random_rk = [real_ranks.index(t) + 1 for t in random_tops]
            mineg_rks.append(np.sum(mineg_rk))
            random_rks.append(np.sum(random_rk))

            mineg_top1 += real_tops[0] in mineg_tops
            random_top1 += real_tops[0] in random_tops

            # print(cd, model_candidates)
            # print(" " * len(cd), real_ranks)

print()
print("MetaG  top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(mineg_counts), np.mean(mineg_rks), mineg_top1/data_count))
print("Fix    top%d count:%.2f | rank:%.2f | top1 recall:%.2f" % (topk, np.mean(random_counts), np.mean(random_rks), random_top1/data_count))
print('------')
print("Full cs size:", len(full_cs))
print("Best if empty ratio: %.2f" % (empty_num/data_count))
print("MetaG  top%d size:%.2f | best recall:%.2f" % (topk_pre, np.mean(mineg_csnum), mineg_best/data_count))
print("Random top%d size:%.2f | best recall:%.2f" % (topk_pre, np.mean(random_csnum), random_best/data_count))
print("Fix    top%d size:%.2f | best recall:%.2f" % (topk_pre, np.mean(fix_csnum), fix_best/data_count))

import matplotlib.pyplot as plt

plt.hist(sims, bins=60, alpha=0.75, color='skyblue')
plt.savefig('./sim_dis_%s.jpg' % task)
plt.show()

"""
CLS, ACC
MetaG top44 count:30.72 | rank:1394.89 | top1 recall:0.78
Fix   top44 count:21.41 | rank:1904.15 | top1 recall:0.50

MetaG top29 count:17.00 | rank:837.60 | top1 recall:0.65
Fix   top29 count:9.67 | rank:1278.42 | top1 recall:0.33

MetaG top8 count:2.51 | rank:181.94 | top1 recall:0.34
Fix   top8 count:0.65 | rank:355.90 | top1 recall:0.08
------
Full cs size: 59
Best if empty ratio: 0.15
MetaG  top6 size:30.18 | best recall:0.66 # 不加入best
MetaG  top6 size:31.17 | best recall:0.66 # 不加入best, thr0.65
MetaG  top6 size:30.16 | best recall:0.66 # 不加入best, def的打分用exp
MetaG  top6 size:27.44 | best recall:0.62 # 相似里面加入最相似的best
Random top6 size:30.99 | best recall:0.53
Fix    top6 size:28.41 | best recall:0.66

MetaG  top7 size:34.52 | best recall:0.72
Random top7 size:34.94 | best recall:0.59
Fix    top7 size:32.12 | best recall:0.74

MetaG  top5 size:26.11 | best recall:0.60 # 不加入best
MetaG  top5 size:23.82 | best recall:0.56 # 相似里面加入最相似的best
Random top5 size:26.99 | best recall:0.47
Fix    top5 size:25.03 | best recall:0.60


10折
MetaG  top6 size:30.04 | best recall:0.66
Random top6 size:31.07 | best recall:0.54
Fix    top6 size:28.57 | best recall:0.68

MetaG  top7 size:34.40 | best recall:0.73
Random top7 size:35.15 | best recall:0.60
Fix    top7 size:31.69 | best recall:0.74


RGS MSE
MetaG top26 count:17.02 | rank:555.60 | top1 recall:0.70
Fix   top26 count:12.79 | rank:698.70 | top1 recall:0.50

MetaG top17 count:8.69 | rank:335.10 | top1 recall:0.56
Fix   top17 count:5.47 | rank:455.11 | top1 recall:0.32


MetaG  top5 count:1.17 | rank:87.70 | top1 recall:0.24
Fix    top5 count:0.50 | rank:134.06 | top1 recall:0.10
------
Full cs size: 45
MetaG  top6 size:31.88 | best recall:0.77 # 不加入best
MetaG  top6 size:29.13 | best recall:0.72 # 加入出现次数最多的best
MetaG  top6 size:29.00 | best recall:0.72 # 相似里面加入最相似的best
Random top6 size:29.54 | best recall:0.68
Fix    top6 size:31.76 | best recall:0.79

MetaG  top5 size:28.29 | best recall:0.70 # 不加入best
Random top5 size:25.50 | best recall:0.60
Fix    top5 size:27.48 | best recall:0.70

"""