import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau
import math

data_dir = './benchmark_data'

perf_data_dict = {
    "CLS":{}, "RGS":{},
}

div_data_dict = {
    "CLS":{}, "RGS":{},
}

sel_ens = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]

for task_type in ["CLS", "RGS"]:
    tt = 3600 if task_type == 'CLS' else 3600
    sub_dir = os.path.join('./benchmark_data/', f'results_{task_type}_{tt}')
    for _result_file in os.listdir(sub_dir):
        result_file = os.path.join(sub_dir, _result_file)

        print(_result_file)
        with open(result_file, 'r') as f:
            results = json.load(f)

        if 'diversity_exp' not in results or results['diversity_exp'] == {}:
            continue

        task_id = _result_file[:-5]

        for size_ratio in results['diversity_exp']:
            dropout = float(size_ratio.split('_')[1])
            if dropout not in sel_ens:
                continue

            if task_id not in perf_data_dict[task_type]:
                perf_data_dict[task_type][task_id] = {}
                div_data_dict[task_type][task_id] = {}

            # 记录base model的质量
            perf, _, diversity = results['diversity_exp'][size_ratio]

            perf_data_dict[task_type][task_id][dropout] = perf
            div_data_dict[task_type][task_id][dropout] = diversity

        tmp_val = list(perf_data_dict[task_type][task_id].values())
        min_p, max_p = np.min(tmp_val), np.max(tmp_val)
        if max_p == min_p:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            continue
        for key in perf_data_dict[task_type][task_id]:
            perf_data_dict[task_type][task_id][key] = (perf_data_dict[task_type][task_id][key] - min_p) / (max_p - min_p)

        tmp_val = list(div_data_dict[task_type][task_id].values())
        min_p, max_p = np.min(tmp_val), np.max(tmp_val)
        if max_p == min_p:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            continue
        for key in div_data_dict[task_type][task_id]:
            div_data_dict[task_type][task_id][key] = (max_p - div_data_dict[task_type][task_id][key]) / (max_p - min_p)


        # 去掉perf下降太快的
        if  perf_data_dict[task_type][task_id][0.05] < 0.5:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)
        # 去掉perf下降太快的
        elif  perf_data_dict[task_type][task_id][0.1] < 0.4:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)
        elif  perf_data_dict[task_type][task_id][0.15] < 0.35:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)
        elif  perf_data_dict[task_type][task_id][0.2] < 0.2:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)
        elif  perf_data_dict[task_type][task_id][0.25] < 0.1:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)
        elif  perf_data_dict[task_type][task_id][0.3] < 0.01:
            perf_data_dict[task_type].pop(task_id)
            div_data_dict[task_type].pop(task_id)
            print("丢弃perf第一下就下降超过0.5的", task_id)

from prettytable import PrettyTable

tar_key = 'ALL'
imp_dict = []
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in perf_data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
        table.add_row(row)
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in sorted(algorithms.keys()):
        num_dict[task_type] = len(algorithms[algorithm])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

        if task_type == tar_key:
            imp_dict.append(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)


if not os.path.exists('./images/diversity/'):
    os.mkdir('./images/diversity/')

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(range(len(imp_dict)), imp_dict, marker='o', linewidth=2.5)
# Customize plot
plt.xlabel('Diversity Weight', fontsize=22)
plt.ylabel('Performance', fontsize=22)
# plt.ylabel('Weight Std', fontsize=22)
# plt.title('Normalized Score of Base Models')
# plt.legend(fontsize=18)
plt.xticks(range(len(sel_ens)), sel_ens, fontsize=20)
plt.tight_layout()

# Show plot
plt.savefig('./images/diversity/perf_with_diversity.png')
plt.show()



imp_dict1 = []
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in div_data_dict.items():

    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
        table.add_row(row)
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in sorted(algorithms.keys()):
        num_dict[task_type] = len(algorithms[algorithm])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

        if task_type == tar_key:
            imp_dict1.append(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(range(len(imp_dict1)), imp_dict1, marker='o', linewidth=2.5)
# Customize plot
plt.xlabel('Diversity Weight', fontsize=22)
plt.ylabel('Diversity', fontsize=22)
# plt.ylabel('Weight Std', fontsize=22)
# plt.title('Normalized Score of Base Models')
# plt.legend(fontsize=18)
plt.xticks(range(len(sel_ens)), sel_ens, fontsize=20)
plt.tight_layout()

# Show plot
plt.savefig('./images/diversity/div_with_diversity.png')
plt.show()


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5.8))
plt.plot(imp_dict1, imp_dict, marker='o', linewidth=3.5)
# Customize plot
plt.xlabel('Diversity', fontsize=22)
plt.ylabel('Performance', fontsize=22)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
# plt.legend(fontsize=18)
plt.tight_layout()

# Show plot
plt.savefig('./images/diversity/perf_with_div.png')
plt.show()
