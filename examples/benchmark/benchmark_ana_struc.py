import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau
import math

data_dir = './benchmark_data'

data_dict = {
    'CLS': {},
    'RGS': {}
}

import pickle as pkl
with open('./images/rank/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)

for task_type in ["CLS", "RGS"]:
    tt = 3600 if task_type == 'CLS' else 3600
    sub_dir = os.path.join('./benchmark_data/', f'results_{task_type}_{tt}')
    for _result_file in os.listdir(sub_dir):
        result_file = os.path.join(sub_dir, _result_file)

        print(_result_file)
        with open(result_file, 'r') as f:
            results = json.load(f)

        if 'struc_exp' not in results or results['struc_exp'] == {}:
            continue

        task_id = _result_file[:-5]

        if task_id not in data_dict[task_type]:
            data_dict[task_type][task_id] = {}

        if task_id not in valid_datasets[task_type]:
            continue

        for size_ratio in results['struc_exp']:
            tmp = size_ratio.split('_')
            ratio, retain = float(tmp[2][7:]), tmp[3][6:]

            # 比较rank
            leader_board =  results['struc_exp'][size_ratio]

            for tmp in leader_board:
                head, layer = tmp.split(': ')[0].split('-')
                head = head.split('_')[0]
                # train-0.81728, train_2--0.26715, test-0.82361, test_2--0.26734, val-0.82394, val_2--0.26430
                tmp_list = tmp.split(': ')[1].split(', ')
                val = float(tmp_list[4].split('val-')[1])
                test = float(tmp_list[2].split('test-')[1])

                data_dict[task_type][task_id][(ratio, retain, head, layer)] = (val, test)



data = {}
kkey = 'Dropout'
data[kkey] = {}
print(kkey)
sel_ens = [0, 0.1, 0.2, 0.3, 0.4] + ['opt']

from prettytable import PrettyTable
from scipy.stats import rankdata
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue

        max_val = -np.inf
        max_test = -np.inf
        scores = []
        for d in sel_ens[:-1]:
            tar = (d, 'True', 'linear', 'L3')
            if tar not in algorithms:
                tar = (d, 'True', 'linear', 'L2')
                if tar not in algorithms:
                    tar = (d, 'True', 'linear', 'L1')

            val, test = algorithms[tar]
            scores.append(test)
            if val > max_val:
                max_val = val
                max_test = test
            elif val == max_val:
                max_test = max(test, max_test)

        scores.append(max_test)

        scores = list(rankdata(-np.array(scores), method='min'))

        row = [task_type, dataset] + scores
        # table.add_row(row)
        for idx, t in enumerate(sel_ens):
            avgs[task_type][t].append(scores[idx])
            avgs["ALL"][t].append(scores[idx])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms.keys():
        num_dict[task_type] = len(algorithms[algorithm])
        std = np.nanstd(algorithms[algorithm]) / math.sqrt(num_dict[task_type])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)
data[kkey] = avg_dict['ALL']


kkey = 'Layer'
data[kkey] = {}
print(kkey)
sel_ens = ['L1', 'L2', 'L3', 'L4', 'L5'] + ['opt']

from prettytable import PrettyTable
from scipy.stats import rankdata
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue

        max_val = -np.inf
        max_test = -np.inf
        scores = []
        for d in sel_ens[:-1]:
            tar = (0.2, 'True', 'linear', d)
            if tar in algorithms:
                val, test = algorithms[tar]
                scores.append(test)
                if val > max_val:
                    max_val = val
                    max_test = test
                elif val == max_val:
                    max_test = max(test, max_test)
            else:
                scores.append(scores[-1])
        scores.append(max_test)

        scores = list(rankdata(-np.array(scores), method='min'))

        row = [task_type, dataset] + scores
        # table.add_row(row)
        for idx, t in enumerate(sel_ens):
            avgs[task_type][t].append(scores[idx])
            avgs["ALL"][t].append(scores[idx])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms.keys():
        num_dict[task_type] = len(algorithms[algorithm])
        std = np.nanstd(algorithms[algorithm]) / math.sqrt(num_dict[task_type])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)
data[kkey] = {f'{idx}': avg_dict['ALL'][f'L{idx}'] for idx in range(1, 6)}
data[kkey]['opt'] = avg_dict['ALL']['opt']



kkey = 'Blender'
data[kkey] = {}
print(kkey)
sel_ens = ['weighted', 'linear', 'lightgbm'] + ['opt']

from prettytable import PrettyTable
from scipy.stats import rankdata
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

count = 0
for task_type, datasets in data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue

        max_val = -np.inf
        max_test = -np.inf
        scores = []
        for d in sel_ens[:-1]:
            if d == 'lightgbm' and task_type == 'RGS':
                d = 'best'
            tar = (0.2, 'True', d, 'L3')
            if tar not in algorithms:
                tar = (0.2, 'True', d, 'L2')
                if tar not in algorithms:
                    tar = (0.2, 'True', d, 'L1')

            val, test = algorithms[tar]
            scores.append(test)
            if val > max_val:
                max_val = val
                max_test = test
            elif val == max_val:
                max_test = max(test, max_test)
        scores.append(max_test)

        scores = list(rankdata(-np.array(scores), method='min'))

        # if task_type == 'RGS':
        #     if count < 20:
        #         count += 1
        #         continue


        row = [task_type, dataset] + scores
        # table.add_row(row)
        for idx, t in enumerate(sel_ens):
            avgs[task_type][t].append(scores[idx])
            avgs["ALL"][t].append(scores[idx])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms.keys():
        num_dict[task_type] = len(algorithms[algorithm])
        std = np.nanstd(algorithms[algorithm]) / math.sqrt(num_dict[task_type])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)
data[kkey] = {
    'ES': avg_dict['ALL']['weighted'],
    'Linear': avg_dict['ALL']['linear'],
    'LGB': avg_dict['ALL']['lightgbm'],
    'opt': avg_dict['ALL']['opt']
}



kkey = 'Retain'
data[kkey] = {}
print(kkey)
sel_ens = ['True', 'False'] + ['opt']

from prettytable import PrettyTable
from scipy.stats import rankdata
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue

        max_val = -np.inf
        max_test = -np.inf
        scores = []
        for d in sel_ens[:-1]:
            tar = (0.2, d, 'linear', 'L3')
            if tar not in algorithms:
                tar = (0.2, d, 'linear', 'L2')
                if tar not in algorithms:
                    tar = (0.2, d, 'linear', 'L1')

            val, test = algorithms[tar]
            scores.append(test)
            if val > max_val:
                max_val = val
                max_test = test
            elif val == max_val:
                max_test = max(test, max_test)
        scores.append(max_test)

        scores = list(rankdata(-np.array(scores), method='min'))

        row = [task_type, dataset] + scores
        # table.add_row(row)
        for idx, t in enumerate(sel_ens):
            avgs[task_type][t].append(scores[idx])
            avgs["ALL"][t].append(scores[idx])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms.keys():
        num_dict[task_type] = len(algorithms[algorithm])
        std = np.nanstd(algorithms[algorithm]) / math.sqrt(num_dict[task_type])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)
data[kkey] = avg_dict['ALL']




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'

# 转换为 DataFrame
records = []
xticks = []
for hp, values in data.items():
    for val, rank in values.items():
        records.append({
            "Hyperparameter": hp,
            "Value": f"{hp}={val}" if val == 'opt' else val,
            "Avg Rank": rank
        })
        xticks.append(r"$\textsc{Opt}$" if val == 'opt' else val)
        # xticks.append(val)

df = pd.DataFrame(records)

# 为了控制柱子顺序，确保 Value 是有序的
df['Group'] = df['Hyperparameter']
df['Value'] = pd.Categorical(df['Value'], categories=df['Value'], ordered=True)

# 设置绘图风格
# sns.set(style="whitegrid")

# 创建图形
plt.figure(figsize=(12, 4))
ax = sns.barplot(data=df, x="Value", y="Avg Rank", hue="Hyperparameter", dodge=False, palette="Set2", width=0.7)

# 添加分组竖线
group_counts = df['Group'].value_counts()[df['Group'].unique()].tolist()
group_boundaries = []
pos = 0
last_indices = []  # 记录每组最后一个柱子的 index
for count in group_counts:
    pos += count
    last_indices.append(pos - 1)
    if pos < len(df):
        group_boundaries.append(pos - 0.5)

for x in group_boundaries:
    ax.axvline(x=x, color='black', linestyle='--', linewidth=1)

# 强调每组最后一个柱子：加粗边框
for i, patch in enumerate(ax.patches):
    if i in last_indices:
        patch.set_edgecolor('black')       # 黑色边框
        patch.set_linewidth(2)             # 边框加粗
        patch.set_zorder(3)                # 提高层级
    else:
        patch.set_edgecolor('none')        # 其他柱子无边框

# 设置图例在图上方（图外）
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=['Dropout ratio', 'Number of layers', 'Blender model', 'Retain or not'],
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    borderaxespad=0.1,
    ncol=len(labels),
    fontsize=20,
    columnspacing=1.7  # 调整这个值来增加列间距
)


# 设置图外边框为黑色
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)  # 可根据需要调整粗细


# 美化图表
plt.xticks(range(len(xticks)), xticks, ha='center', fontsize=18)

# xticks = ax.get_xticklabels()
# # 将前五个刻度标签旋转 30 度
# for i, tick in enumerate(xticks):  # 前五个标签
#     tick.set_rotation(30)  # 旋转 30 度

plt.ylim(0.8, 3.5)
plt.yticks([1, 2, 3], fontsize=20)
plt.ylabel("Average Test Rank", fontsize=22)
plt.xlabel("Hyperparameter Value", fontsize=22)
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.18)
plt.grid(False)
# 显示图表
if not os.path.exists('./images/struc'):
    os.mkdir('./images/struc')
plt.savefig('./images/struc/struc.pdf')
plt.savefig('./images/struc/struc.png')
plt.show()
