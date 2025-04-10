import numpy as np
import pickle as pkl
import pandas as pd
from scipy.stats import spearmanr

cls_val_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_algo2perf_valcor.pkl'
cls_test_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_algo2perf_testcor.pkl'

rgs_val_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_algo2perf_valcor.pkl'
rgs_test_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_algo2perf_testcor.pkl'
cls_tasks = None
cls_algo = None
rgs_tasks = None
rgs_algo = None

with open(cls_val_path, 'rb') as f:
    res = pkl.load(f)
    cls_val = res['perf4algo']['acc']
    cls_tasks = res['task_ids']
    cls_algo = res['algorithms_included']
with open(cls_test_path, 'rb') as f:
    cls_test = pkl.load(f)['perf4algo']['acc']


with open(rgs_val_path, 'rb') as f:
    res = pkl.load(f)
    rgs_val = res['perf4algo']['mse']
    rgs_tasks = res['task_ids']
    rgs_algo = res['algorithms_included']

with open(rgs_test_path, 'rb') as f:
    rgs_test = pkl.load(f)['perf4algo']['mse']


cls_corrs = []
cls_sps = []
cls_val_ranks = []
cls_test_ranks = []
for i in range(len(cls_val)):
    tmp1 = cls_val[i]
    tmp2 = cls_test[i]
    mask = np.isfinite(tmp1) & np.isfinite(tmp2)
    if np.all(mask == 0):
        continue
    cls_val_ranks.append(pd.Series(tmp1).rank(ascending=False).values)
    cls_test_ranks.append(pd.Series(tmp2).rank(ascending=False).values)
    tmp1 = tmp1[mask]
    tmp2 = tmp2[mask]
    corr = np.corrcoef(tmp1, tmp2)[0, 1]
    s, _ = spearmanr(tmp1, tmp2)
    cls_sps.append(s)
    if np.isnan(corr):
        continue
    cls_corrs.append(corr)

rgs_corrs = []
rgs_sps = []
rgs_val_ranks = []
rgs_test_ranks = []
for i in range(len(rgs_val)):
    tmp1 = rgs_val[i]
    tmp2 = rgs_test[i]
    mask = np.isfinite(tmp1) & np.isfinite(tmp2)
    if np.all(mask == 0):
        continue
    if 'cpu_act' in rgs_tasks[i]:
        rgs_val_ranks.append(pd.Series(tmp1).rank(ascending=False).values)
        rgs_test_ranks.append(pd.Series(tmp2).rank(ascending=False).values)
    tmp1 = tmp1[mask]
    tmp2 = tmp2[mask]
    corr = np.corrcoef(tmp1, tmp2)[0, 1]
    s, _ = spearmanr(tmp1, tmp2)
    rgs_sps.append(s)
    if np.isnan(corr):
        continue
    rgs_corrs.append(corr)

print(np.mean(cls_corrs), np.nanmean(cls_sps))
print(np.mean(rgs_corrs), np.nanmean(rgs_sps))
cls_val_rank = np.round(np.mean(cls_val_ranks, axis=0), 2)
cls_test_rank = np.round(np.mean(cls_test_ranks, axis=0), 2)
rgs_val_rank = np.round(np.mean(rgs_val_ranks, axis=0), 2)
rgs_test_rank = np.round(np.mean(rgs_test_ranks, axis=0), 2)

cls_idx = np.argsort(cls_corrs)
rgs_idx = np.argsort(rgs_corrs)

top = 10

for i in range(top):
    idx = cls_idx[i]
    print(cls_corrs[idx], cls_tasks[idx])

print("---------")

for i in range(top):
    idx = rgs_idx[i]
    print(rgs_corrs[idx], rgs_tasks[idx])

# 看看xgboost的平均排名

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "set"] + cls_algo
table.field_names = headers
table.add_row(['CLS', 'Val'] + list(cls_val_rank))
table.add_row(['CLS', 'Test'] + list(cls_test_rank))
print(table)


table = PrettyTable()
headers = ["Task Type", "set"] + rgs_algo
table.field_names = headers
table.add_row(['RGS', 'Val'] + list(rgs_val_rank))
table.add_row(['RGS', 'Test'] + list(rgs_test_rank))
print(table)
"""
三个的中位数
0.6743433871421664 0.6161536147944796
0.4262022523663263 0.4432447204423183
只用第一个

"""
breakpoint()