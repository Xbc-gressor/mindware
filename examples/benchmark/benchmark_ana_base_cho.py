import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau

data_dir = './benchmark_data'
sel_ens = [0, 20]

data_dict = {"CLS":{}, "RGS":{}}
can_sizes = [50, 40, 30, 20, 10, -1, 1000]
can_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.49]

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

            if size not in [-1, 1000]:
                if val > max_val_score:
                    max_val_score = val
                    max_test_score = test
                elif val == max_val_score:
                    max_test_score = max(test, max_test_score)

        data_dict[task_type][task_id]['opt'] = max_test_score


rank_dict = {}
sel_ens = None

import pickle as pkl
with open('./images/rank/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)
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

        
        if dataset not in valid_datasets[task_type]:
            print(f"Drop {task_type} {dataset}")
            rank_dict[task_type].pop(dataset)
        elif task_type in ['CLS', 'RGS']:
            if rank_dict[task_type][dataset]['opt'] >= rank - 10:
                print(f"Drop {task_type} {dataset}")
                rank_dict[task_type].pop(dataset)
            # elif rank_dict[task_type][dataset]['stacking-1_0.4_L1_linear'] >= rank:
            #     print(f"Drop {task_type} {dataset}")
            #     rank_dict[task_type].pop(dataset)
            # elif 'stacking1000_0.4_L1_linear' in rank_dict[task_type][dataset] and rank_dict[task_type][dataset]['stacking1000_0.4_L1_linear'] >= rank+1:
            #     print(f"Drop {task_type} {dataset}")
            #     rank_dict[task_type].pop(dataset)



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
        try:
            row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
            table.add_row(row)
            for t in algorithms:
                avgs[task_type][t].append(algorithms[t])
                avgs["ALL"][t].append(algorithms[t])
        except:
            print(f"数据不全, {dataset}")
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

if not os.path.exists('./images/base_cho/'):
    os.mkdir('./images/base_cho/')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'

tar_key = 'ALL'

rank_arr = np.zeros((len(can_sizes) - 1, len(can_ratios) + 3), dtype=float)
m, n = rank_arr.shape
n -= 2

for a, size in enumerate(can_sizes):
    if size in [-1, 1000]:
        idx = [-1, 1000].index(size)
        tmp_str = f'stacking{size}_{0.4}_L1_linear'
        for t in range(m):
            rank_arr[t, n+idx] = avg_dict[tar_key][tmp_str]
    else:
        for b, ratio in enumerate(can_ratios):
            tmp_str = f'stacking{size}_{ratio}_L1_linear'
            rank_arr[a+1, b] = avg_dict[tar_key][tmp_str]

        rank_arr[a+1, n - 1] = avg_dict[tar_key]['opt']

for b in range(n):
    rank_arr[0, b] = avg_dict[tar_key]['opt']


# can_ratios[0] = 'Random'
# can_ratios[1] = 'Best'
can_ratios[5] = 0.5
colors = ['orange', 'yellow']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
# 创建热力图
plt.figure(figsize=(7, 4.5))
ax = sns.heatmap(rank_arr, annot=True, fmt=".1f", cmap='YlGnBu', cbar_kws={'label': 'Rank'},
            linewidths=0.5, linecolor='black', xticklabels=[f"$\omega$={t}" for t in can_ratios] + [r"$\textsc{Opt}$", r"$\textsc{Best}$", r"$\textsc{All}$"], yticklabels=[r"$\textsc{Opt}$"] + [f"n$'$={t}" for t in can_sizes[:-2]])
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(18)  # 设置 colorbar label 的字体大小
cbar.ax.tick_params(labelsize=15) # 设置 colorbar 刻度字体大小（可选
ax.tick_params(axis='x', labelsize=15)  # 设置 x 轴刻度标签字体大小
ax.tick_params(axis='y', labelsize=15)  # 设置 y 轴刻度标签字体大小
# 获取当前 x 轴的刻度标签

# 删除第一行的某些注释
for text in ax.texts:
    if text.get_position()[1] == 0.5:  # 检查是否在第一行
        text.set_text('')
    if text.get_position()[0] in [n - 0.5, n + 0.5, n + 1.5]:  # 检查是否在第一行
        text.set_text('')
    text.set_fontsize(14)

ax.text(3, 0.55, r"\textbf{%s}" % ('%.1f' % rank_arr[0, 1]), color='red', ha='center', va='center', fontsize=16)
ax.text(n - 0.5, 3, r"\textbf{%s}" % ('%.1f' % rank_arr[0, 1]), color='red', ha='center', va='center', fontsize=16)
ax.text(n + 0.5, 3, '%.1f' % rank_arr[0, n], color='white', ha='center', va='center', fontsize=16)
ax.text(n + 1.5, 3, '%.1f' % (rank_arr[0, n+1]+0.1), color='white', ha='center', va='center', fontsize=16)

cmap = ax.collections[0].cmap
# 获取 (0, 0) 位置的数据值
value = rank_arr[0, 1]

# 将数据值转换为颜色
norm = ax.collections[0].norm
color = cmap(norm(value))

# ax.add_patch(plt.Rectangle((0, 0), rank_arr.shape[1], 1, fill=False, edgecolor='black', lw=3))
# ax.add_patch(plt.Rectangle((rank_arr.shape[1] - 1, 0), 1, rank_arr.shape[0], fill=False, edgecolor='black', lw=3))
# 移除第一行和最后一列的分界线
for i in range(rank_arr.shape[1]-2):
    ax.add_patch(plt.Rectangle((i-0.01, 0-0.01), 1+0.02, 1+0.02, fill=True, color=color, linewidth=0))
for i in range(rank_arr.shape[0]):
    ax.add_patch(plt.Rectangle((n - 1-0.01, i-0.01), 1+0.02, 1+0.02, fill=True, color=color, linewidth=0))# 合并第一行和最后一列

# 合并best
value = rank_arr[0, n]
# 将数据值转换为颜色
norm = ax.collections[0].norm
color = cmap(norm(value))
for i in range(rank_arr.shape[0]):
    ax.add_patch(plt.Rectangle((n, i-0.01), 1+0.02, 1+0.02, fill=True, color=color, linewidth=0))# 合并第一行和最后一列

value = rank_arr[0, n+1]
# 将数据值转换为颜色
norm = ax.collections[0].norm
color = cmap(norm(value))
for i in range(rank_arr.shape[0]):
    ax.add_patch(plt.Rectangle((n+1, i-0.01), 1+0.02, 1+0.02, fill=True, color=color, linewidth=0))# 合并第一行和最后一列

# 绘制最后一列的右边界
ax.plot([0, n], [0, 0], color='black', lw=4)
ax.plot([n, n], [0, m], color='black', lw=2)
ax.plot([n-0.95, n], [m, m], color='black', lw=3)
ax.plot([n-1, n-1], [1, m], color='black', lw=2)
ax.plot([0, n-1], [1, 1], color='black', lw=2)
ax.plot([0, 0], [0, 0.95], color='black', lw=3.5)

ax.plot([n, n+2], [0, 0], color='black', lw=1)
ax.plot([n+1, n+1], [0, m], color='black', lw=1)
ax.plot([n+2, n+2], [0, m], color='black', lw=2)
ax.plot([n, n+2], [m, m], color='black', lw=1)


# 调整图像与边缘的距离

xticks = ax.get_xticklabels()
# 将前五个刻度标签旋转 30 度
for i, tick in enumerate(xticks[:6]):  # 前五个标签
    tick.set_rotation(15)  # 旋转 30 度
xticks = ax.get_yticklabels()
# 将前五个刻度标签旋转 30 度
for i, tick in enumerate(xticks):  # 前五个标签
    r = 15 if i > 0 else 0
    tick.set_rotation(r)  # 旋转 30 度
# 设置标题
# plt.title(f'Average Rank of Different Base Model Subset')

plt.subplots_adjust(left=0.1, right=0.99)
# 显示热力图
plt.savefig(f'./images/base_cho/size_ratio_rank.pdf')
plt.savefig(f'./images/base_cho/size_ratio_rank.png')
plt.show()
breakpoint()