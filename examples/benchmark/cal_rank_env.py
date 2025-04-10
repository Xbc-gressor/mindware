import re
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info


rank_fields = [
    # 'cashfe-block_1-norefit-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection50-filter_m6_p6-best',
    # 'cashfe-block_1-compress-filter_m6_p6-ens', 
    'cashfe-block_1-ensemble_selection50-filter_m6_p6-ens', 
   'cashfe-block_1-ensemble_selection40-filter_m6_p6-ens',
   'cashfe-block_1-ensemble_selection30-filter_m6_p6-ens',
   'cashfe-block_1-ensemble_selection20-filter_m6_p6-ens',
   'cashfe-block_1-ensemble_selection10-filter_m6_p6-ens',
   'cashfe-block_1-blending20-filter_m6_p6-ens',
   'cashfe-block_1-blending10-filter_m6_p6-ens',
   'cashfe-block_1-blending5-filter_m6_p6-ens'
]

rank_fields = [
    'autogluon-ens',
    'cashfe-block_1-ensemble_selection50-filter_m6_p6-best',

    'cashfe-block_1-ensemble_selection50-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection40-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection30-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection20-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection10-filter_m6_p6-ens',
    'cashfe-block_1-ensemble_selection5-filter_m6_p6-ens',

    'cashfe-block_1-blendingL50_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL50_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL50_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL50_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL50_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL50_0.49-filter_m6_p6-ens',

    'cashfe-block_1-blendingL40_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL40_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL40_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL40_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL40_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL40_0.49-filter_m6_p6-ens',

    'cashfe-block_1-blendingL30_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL30_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL30_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL30_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL30_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL30_0.49-filter_m6_p6-ens',

    'cashfe-block_1-blendingL20_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL20_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL20_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL20_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL20_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL20_0.49-filter_m6_p6-ens',

    'cashfe-block_1-blendingL10_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL10_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL10_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL10_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL10_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL10_0.49-filter_m6_p6-ens',

    'cashfe-block_1-blendingL5_0.0-filter_m6_p6-ens',
    'cashfe-block_1-blendingL5_0.1-filter_m6_p6-ens',
    'cashfe-block_1-blendingL5_0.2-filter_m6_p6-ens',
    'cashfe-block_1-blendingL5_0.3-filter_m6_p6-ens',
    'cashfe-block_1-blendingL5_0.4-filter_m6_p6-ens',
    'cashfe-block_1-blendingL5_0.49-filter_m6_p6-ens'
]

# cashfe-block_1-filter_m6_p6|cashfe-block_1-ensemble_selection50-filter_m6_p6_ens|cashfe-block_1-ensemble_selection40-filter_m6_p6_ens|cashfe-block_1-ensemble_selection50-filter_m6_p6_ens

def parse_data(file_path):
    results = {
        "CLS": {},
        "RGS": {},
    }
    # "cashfe_best": [], "cashfe_ens": [], "cash_best": [], "cash_ens": []

    with open(file_path, 'r') as file:
        for line in file:
            if '---------------' in line:
                break
        for line in file:  # Continue reading after the '---------------'
            match = re.match(r"(CLS|RGS): ([^,]+), (\w+): (-?\d+\.\d+), (-?\d+\.\d+)", line)
            if match:
                task_type, algorithm, dataset, best, ens = match.groups()
                if dataset not in results[task_type]:
                    results[task_type][dataset] = {}
                results[task_type][dataset][f"{algorithm}-best"] = float(best)
                results[task_type][dataset][f"{algorithm}-ens"] = float(ens) 

    ranks = {
        "CLS": {},
        "RGS": {},
    }
    for task_type in results:
        for dataset in results[task_type]:
            if dataset not in ranks[task_type]:
                ranks[task_type][dataset] = {}
                
            datas = {key: value for key, value in results[task_type][dataset].items() if key in rank_fields}
            if len(datas) < len(rank_fields):
                print(f"Missing data for {task_type} {dataset}", datas)
                continue
            # Sort algorithms based on scores, higher scores get higher ranks
            sorted_scores = sorted(datas.items(), key=lambda x: x[1], reverse=True)
            rankings = {}
            current_rank = 1
            items_at_rank = 0  # 追踪当前排名的项数

            # 处理第一个元素
            previous_value = sorted_scores[0][1]
            rankings[sorted_scores[0][0]] = current_rank
            items_at_rank += 1

            # 遍历排序后的元素，从第二个开始
            for name, value in sorted_scores[1:]:
                if value == previous_value:
                    # 如果当前值与前一个值相同，则使用相同的排名
                    rankings[name] = current_rank
                    items_at_rank += 1
                else:
                    # 如果当前值不同，则更新排名，排名应该是当前排名加上当前排名项的数量
                    current_rank += items_at_rank
                    rankings[name] = current_rank
                    previous_value = value
                    items_at_rank = 1  # 重置当前排名的项数
        
            # Assign ranks
            ranks[task_type][dataset] = rankings

    return results, ranks

def calculate_averages(results):
    for task_type in results:
        for algorithm in results[task_type]:
            avg_best = sum(results[task_type][algorithm]["best"]) / len(results[task_type][algorithm]["best"])
            avg_ens = sum(results[task_type][algorithm]["ens"]) / len(results[task_type][algorithm]["ens"])
            print(f"{task_type} {algorithm} average best: {avg_best}")
            print(f"{task_type} {algorithm} average ens: {avg_ens}")

# Replace 'path_to_your_file.txt' with the actual path to your data file
file_path = './res_refit_data.txt'
results, ranks = parse_data(file_path)

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"]
for tmp in rank_fields:
    if 'autogluon' in tmp:
        headers.append('autogluon')
    elif 'best' in tmp:
        headers.append('best')
    elif 'compress' in tmp:
        headers.append("compress")
    else:
        print(tmp)
        tmp_list = tmp.split('-')
        tmp_list = [tmp_list[2], tmp_list[4]]
        headers.append('-'.join(tmp_list))
avgs = {
    "CLS": {t:[] for t in rank_fields},
    "RGS": {t:[] for t in rank_fields},
    "ALL": {t:[] for t in rank_fields}
}
top_counts = {
    1: {
        "CLS": {t:[] for t in rank_fields},
        "RGS": {t:[] for t in rank_fields},
        "ALL": {t:[] for t in rank_fields}
    },
    2: {
        "CLS": {t:[] for t in rank_fields},
        "RGS": {t:[] for t in rank_fields},
        "ALL": {t:[] for t in rank_fields}
    },
    3: {
        "CLS": {t:[] for t in rank_fields},
        "RGS": {t:[] for t in rank_fields},
        "ALL": {t:[] for t in rank_fields}
    },
}

for task_type, datasets in ranks.items():
    
    table.field_names = headers
    
    dataset_names = None
    if task_type == "CLS":
        dataset_names = [n for n in cls_info.index if n in datasets]
    else:
        dataset_names = [n for n in rgs_info.index if n in datasets]
    
    # 填充表格行数据
    for dataset in dataset_names:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        row = [task_type, dataset] + [algorithms[t] for t in rank_fields]
        table.add_row(row)
        
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(rank_fields))

for task_type, algorithms in avgs.items():
    for k in top_counts.keys():
        top_counts[k][task_type] = {}
    for algorithm in algorithms:
        for k in top_counts.keys():
            top_counts[k][task_type][algorithm] = len(np.where(np.array(algorithms[algorithm]) <= k)[0])
        algorithms[algorithm] = np.mean(algorithms[algorithm])

    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in rank_fields])

print(table)
import csv
with open('./images/ensrank.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(table.field_names)
    # 写入数据行
    for row in table.rows:
        writer.writerow(row)


import seaborn as sns
import matplotlib.pyplot as plt
sizes = [5, 10, 20, 30, 40, 50]
ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.49]

for key, values in avgs.items():
    rank_arr = np.zeros((6, 9))
    for mth, r in values.items():
        if 'autogluon' in mth:
            rank_arr[:, 0] = r
            continue
        if 'best' in mth:
            rank_arr[:, 1] = r
            continue
        ens = mth.split('-')[2]
        if 'ensemble_selection' in ens:
            size = int(ens[len('ensemble_selection'):])
            rank_arr[sizes.index(size), 2] = r
        else:
            tmp = ens[len('blendingL'):]
            size, ratio = tmp.split('_')
            size = int(size)
            ratio = float(ratio)
            rank_arr[sizes.index(size), ratios.index(ratio)+3] = r
    # 自定义标签
    x_labels = ['gluon', 'best', 'sel'] + ratios
    y_labels = sizes

    # 创建热力图
    plt.figure(figsize=(6, 4))
    sns.heatmap(rank_arr, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Rank'},
                linewidths=0.5, linecolor='black', xticklabels=x_labels, yticklabels=y_labels)
    plt.xticks(rotation=30)
    # 设置标题
    plt.title(f'Model Rank Heatmap({key})')

    # 显示热力图
    plt.savefig(f'./images/ensrank_{key}.png')
    plt.show()
    
for topk, _top_counts in top_counts.items():
    for key, values in _top_counts.items():
        rank_arr = np.zeros((6, 9))
        for mth, r in values.items():
            if 'autogluon' in mth:
                rank_arr[:, 0] = r
                continue
            if 'best' in mth:
                rank_arr[:, 1] = r
                continue
            ens = mth.split('-')[2]
            if 'ensemble_selection' in ens:
                size = int(ens[len('ensemble_selection'):])
                rank_arr[sizes.index(size), 2] = r
            else:
                tmp = ens[len('blendingL'):]
                size, ratio = tmp.split('_')
                size = int(size)
                ratio = float(ratio)
                rank_arr[sizes.index(size), ratios.index(ratio)+3] = r
        # 自定义标签
        x_labels = ['gluon', 'best', 'sel'] + ratios
        y_labels = sizes

        # 创建热力图
        plt.figure(figsize=(6, 4))
        sns.heatmap(rank_arr, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Rank'},
                    linewidths=0.5, linecolor='black', xticklabels=x_labels, yticklabels=y_labels)
        plt.xticks(rotation=30)
        # 设置标题
        plt.title(f'Top {topk} Count Heatmap({key})')

        # 显示热力图
        plt.savefig(f'./images/top{topk}count_{key}.png')
        plt.show()
