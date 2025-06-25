import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau

data_dir = './benchmark_data'
sel_ens = [0, 20]

data_dict = {"CLS":{}, "RGS":{}}
can_sizes = [40, 30, 20, 10, 5]
can_ratios = [-1, 0, 0.1, 0.2, 0.3, 0.4]

for task_type in ["CLS", "RGS"]:
    tt = 3600 if task_type == 'CLS' else 3600
    sub_dir = os.path.join('./benchmark_data/', f'results_{task_type}_{tt}')
    for _result_file in os.listdir(sub_dir):
        result_file = os.path.join(sub_dir, _result_file)

        print(_result_file)
        with open(result_file, 'r') as f:
            results = json.load(f)

        # if "fixed_ens" not in results:
        #     continue
        # if isinstance(results["fixed_ens"], list):
        #     results["fixed_ens"] = results["fixed_ens"][0]
        # with open(result_file, 'w') as f:
        #     json.dump(results, f, indent=4)

        if 'size_exp' not in results or results['size_exp'] == {}:
            continue

        task_id = _result_file[:-5]
        if task_id not in data_dict[task_type]:
            data_dict[task_type][task_id] = {}

        max_val_score = -np.inf
        max_test_score = -np.inf
        for size_ratio in results['size_exp']:
            tmp = size_ratio.split('_')
            size, ratio = int(tmp[0][8:]), float(tmp[1])

            if size not in can_sizes or ratio not in can_ratios:
                continue

            tmp = results['size_exp'][size_ratio][0].split(', ')
            test = float(tmp[2].split('test-')[1])
            val = float(tmp[4].split('val-')[1])
            data_dict[task_type][task_id][size_ratio] = test

            if val > max_val_score:
                max_val_score = val
                max_test_score = test
            elif val == max_val_score:
                max_test_score = max(test, max_test_score)

        data_dict[task_type][task_id]['opt'] = max_test_score


rank_dict = {}
sel_ens = None

for task_type, datasets in data_dict.items():
    if task_type not in rank_dict:
        rank_dict[task_type] = {}
    for dataset, size_ratios in datasets.items():
        if dataset not in rank_dict[task_type]:
            rank_dict[task_type][dataset] = {}

        if sel_ens is None:
            sel_ens = list(size_ratios.keys())

        # 根据值对键进行排序
        sorted_items = sorted(size_ratios.items(), key=lambda x: -x[1])

        # 创建一个字典来存储排名
        rank = 1
        # 遍历排序后的项目，为每个键分配排名
        for i, (key, value) in enumerate(sorted_items):
            if i > 0 and value != sorted_items[i - 1][1]:
                rank = i + 1
            rank_dict[task_type][dataset][key] = rank
    
        if task_type in ['CLS', 'RGS']:
            if rank_dict[task_type][dataset]['opt'] >= rank - 10:
                print(f"Drop {task_type} {dataset}")
                rank_dict[task_type].pop(dataset)


from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in rank_dict.items():

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
    for algorithm in algorithms:
        num_dict[task_type] = len(algorithms[algorithm])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

tar_key = 'ALL'

rank_arr = np.zeros((len(can_sizes) + 1, len(can_ratios) + 1), dtype=float)
m, n = rank_arr.shape

for a, size in enumerate(can_sizes):
    for b, ratio in enumerate(can_ratios):
        tmp_str = f'stacking{size}_{ratio}_L1_linear'
        rank_arr[a+1, b] = avg_dict[tar_key][tmp_str]

    rank_arr[a+1, n - 1] = avg_dict[tar_key]['opt']

for b in range(n):
    rank_arr[0, b] = avg_dict[tar_key]['opt']


can_ratios[0] = 'Random'
can_ratios[1] = 'Best'
colors = ['orange', 'yellow']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
# 创建热力图
plt.figure(figsize=(5.5, 4.5))
ax = sns.heatmap(rank_arr, annot=True, fmt=".1f", cmap='YlGnBu', cbar_kws={'label': 'Rank'},
            linewidths=0.5, linecolor='black', xticklabels=can_ratios + ['opt'], yticklabels=['opt'] + can_sizes)

# 删除第一行的某些注释
for text in ax.texts:
    if text.get_position()[1] == 0.5:  # 检查是否在第一行
        text.set_text('')
    if text.get_position()[0] == n - 0.5:  # 检查是否在第一行
        text.set_text('')
    text.set_fontsize(10)

ax.text(3.2, 0.5, '%.1f' % rank_arr[0, 1], color='black', ha='center', va='center', fontweight='bold', fontsize=11)
ax.text(n - 0.5, 3, '%.1f' % rank_arr[0, 1], color='black', ha='center', va='center', fontweight='bold', fontsize=11)

cmap = ax.collections[0].cmap
# 获取 (0, 0) 位置的数据值
value = rank_arr[0, 1]

# 将数据值转换为颜色
norm = ax.collections[0].norm
color = cmap(norm(value))

# ax.add_patch(plt.Rectangle((0, 0), rank_arr.shape[1], 1, fill=False, edgecolor='black', lw=3))
# ax.add_patch(plt.Rectangle((rank_arr.shape[1] - 1, 0), 1, rank_arr.shape[0], fill=False, edgecolor='black', lw=3))
# 移除第一行和最后一列的分界线
for i in range(rank_arr.shape[1]):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, fill=True, color=color, linewidth=0))
for i in range(rank_arr.shape[0]):
    ax.add_patch(plt.Rectangle((n - 1, i), 1, 1, fill=True, color=color, linewidth=0))# 合并第一行和最后一列

# 绘制最后一列的右边界
ax.plot([0, n], [0, 0], color='black', lw=4)
ax.plot([n, n], [0, m], color='black', lw=4)
ax.plot([n-0.95, n], [m, m], color='black', lw=3)
ax.plot([n-1, n-1], [1, m], color='black', lw=2)
ax.plot([0, n-1], [1, 1], color='black', lw=2)
ax.plot([0, 0], [0, 0.95], color='black', lw=3.5)



# 调整图像与边缘的距离
plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1)
plt.xticks(rotation=30)
# 设置标题
plt.title(f'Size Ratio Ranks of {tar_key}')

# 显示热力图
plt.savefig(f'./images/size_ratio_rank.png')
plt.show()
breakpoint()