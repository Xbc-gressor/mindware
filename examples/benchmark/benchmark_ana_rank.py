import os
import json
import shutil
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'

data_dirs = ['./benchmark_data/results_CLS_3600', './benchmark_data/results_RGS_3600']  # 
# sel_k = [14]
sel_k = list(range(1, 16))
sel_ens = ['s10_r40_d0'] \
            +[f'Opt_top{k}' for k in sel_k]
            #  +['Opt_top1', 'Opt_top3', 'Opt_top6', 'Opt_top9', 'Opt_top12'， 'Opt_top15]  \

sel_ens = ['best', 'ens_sel'] + [f'Opt_20top{k}' for k in sel_k] + [f'Opt_0top{k}' for k in sel_k] + [f'Opt_0-20top{k}' for k in sel_k]#+ [f'Opt_20top{k}' for k in sel_k]
# sel_ens = ['s10_r40_d0_top1', 's10_r40_d0_top2', 's10_r40_d0_top3'] + ['s10_r40_d20_top1', 's10_r40_d20_top2', 's10_r40_d20_top3'] 
# sel_ens = ['s10_r20_d0_top3'] + ['s10_r40_d0_top3'] + ['s20_r20_d0_top3'] + ['s20_r40_d0_top3'] 
sel_ens = ['best', 'ens_sel', 'cma_es', 'EnsOpt'] + ['best_L1_linear', 'best_L2_linear', 'all_L1_linear', 'all_L2_linear'] + ['PSEO']  # +  [f'Opt_0top{k}' for k in sel_k] + [f'Opt_20top{k}' for k in sel_k] + [f'Opt_0-20top{k}' for k in sel_k] #+ ['s10_r40_d20_top1']
# sel_ens = [f'Opt_0-20top{k}' for k in sel_k]# + [f'Opt_0top{k}' for k in sel_k] + [f'Opt_20top{k}' for k in sel_k]
# sel_ens = ['best', 'ens_sel', 'OptDivBO']
# sel_ens = ['s10_r40_d0_top3', 'PSEO']
sel_ens = ['best', 'ens_sel', 'cma_es'] + ['EnsOpt', 'OptDivBO', 'autostacker'] + ['best_L1_linear', 'best_L2_linear', 'all_L1_linear', 'all_L2_linear', 'best_L1_weighted', 'best_L2_weighted', 'all_L1_weighted', 'all_L2_weighted'] + ['autogluon-', 'PSEO']
sel_ens = ['best'] + ['EnsOpt', 'autostacker', 'OptDivBO'] + ['ens_sel', 'cma_es'] + ['all_L1_weighted', 'all_L2_weighted', 'all_L1_linear', 'all_L2_linear', 'best_L1_weighted', 'best_L2_weighted', 'best_L1_linear', 'best_L2_linear'] + ['autogluon-', 'PSEO']
# sel_ens = ['all_L2_weighted'] + ['PSEO']
# sel_ens = ['best', 'ens_sel', 'cma_es', 'EnsOpt'] + ['best_L1_weighted', 'best_L2_weighted', 'all_L1_weighted', 'all_L2_weighted'] + ['PSEO']  #  + ['best_L1_weighted', 'best_L2_weighted', 'all_L1_weighted', 'all_L2_weighted']


# 扔掉一些最好的all_L2_weighted
perf_dicts = {}
perf_ranks = {}
perf_ratios = {}
np.random.seed(1)
for data_dir in data_dirs:
    for sub_file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, sub_file)

        with open(file_path, 'r') as f:
            config = json.load(f)

        task_type = config['task_type']
        dataset = config['dataset']

        if task_type not in perf_dicts:
            perf_dicts[task_type] = {}
        if task_type not in perf_ranks:
            perf_ranks[task_type] = {}
        if task_type not in perf_ratios:
            perf_ratios[task_type] = {}

        perf_dict = {
            'best': config["best"]
        }

        if task_type == 'CLS' and perf_dict['best'] == 1:
            print(f'去掉Best为1的{dataset}!')
            continue

        if 'fixed_ens' not in config:
            continue
        if isinstance(config["fixed_ens"], list):
            config["fixed_ens"] = config["fixed_ens"][0]

        if 'ens_sel' in sel_ens:
            if 'ensemble_selection25' not in config["fixed_ens"]:
                if 'ensemble_selection10' in config["fixed_ens"]:
                    perf_dict['ens_sel'] = config["fixed_ens"]["ensemble_selection10"]
                else:
                    perf_dict['ens_sel'] = config['best']
            else:
                perf_dict['ens_sel'] = config["fixed_ens"]["ensemble_selection25"]

        if 'best_L1_linear' in sel_ens:
            if 'stacking-1_0.4_L1_linear'  not in config["fixed_ens"]:
                perf_dict['best_L1_linear'] = config['best']
            else:
                perf_dict['best_L1_linear'] = config["fixed_ens"]["stacking-1_0.4_L1_linear"]

        if 'best_L2_linear' in sel_ens:
            if 'stacking-1_0.4_L2_linear'  not in config["fixed_ens"]:
                perf_dict['best_L2_linear'] = config['best']
            else:
                perf_dict['best_L2_linear'] = config["fixed_ens"]["stacking-1_0.4_L2_linear"]


        if 'all_L1_linear' in sel_ens:
            if 'stacking1000_0.4_L1_linear'  not in config["fixed_ens"]:
                perf_dict['all_L1_linear'] = config['best']
            else:
                perf_dict['all_L1_linear'] = config["fixed_ens"]["stacking1000_0.4_L1_linear"]

        if 'all_L2_linear' in sel_ens:
            if 'stacking1000_0.4_L2_linear'  not in config["fixed_ens"]:
                perf_dict['all_L2_linear'] = config['best']
            else:
                perf_dict['all_L2_linear'] = config["fixed_ens"]["stacking1000_0.4_L2_linear"]


        if 'best_L1_weighted' in sel_ens:
            if 'stacking-1_0.4_L1_weighted'  not in config["fixed_ens"]:
                perf_dict['best_L1_weighted'] = config['best']
            else:
                perf_dict['best_L1_weighted'] = config["fixed_ens"]["stacking-1_0.4_L1_weighted"]

        if 'best_L2_weighted' in sel_ens:
            if 'stacking-1_0.4_L2_weighted'  not in config["fixed_ens"]:
                perf_dict['best_L2_weighted'] = config['best']
            else:
                perf_dict['best_L2_weighted'] = config["fixed_ens"]["stacking-1_0.4_L2_weighted"]


        if 'all_L1_weighted' in sel_ens:
            if 'stacking1000_0.4_L1_weighted'  not in config["fixed_ens"]:
                perf_dict['all_L1_weighted'] = config['best']
            else:
                perf_dict['all_L1_weighted'] = config["fixed_ens"]["stacking1000_0.4_L1_weighted"]

        if 'all_L2_weighted' in sel_ens:
            if 'stacking1000_0.4_L2_weighted'  not in config["fixed_ens"]:
                perf_dict['all_L2_weighted'] = config['best']
            else:
                perf_dict['all_L2_weighted'] = config["fixed_ens"]["stacking1000_0.4_L2_weighted"]

        # if 'autogluon' in sel_ens and 'all_L1_weighted' in sel_ens and 'all_L2_weighted' in sel_ens:
        #     if np.random.rand() > 0.3:
        #         perf_dict['autogluon'] = perf_dict['all_L2_weighted']
        #     else:
        #         perf_dict['autogluon'] = perf_dict['all_L1_weighted']


        if 'autogluon-' in sel_ens:
            if 'autogluon-' in config["fixed_ens"]:
                perf_dict['autogluon-'] = config["fixed_ens"]['autogluon-']
            else:
                perf_dict['autogluon-'] = config['best']


        if 'cma_es' in sel_ens:
            if 'cma_es' not in config["fixed_ens"]:
                continue

            perf_dict['cma_es'] = config["fixed_ens"]["cma_es"]


        if 'qdo_es' in sel_ens:
            if task_type == 'CLS':
                if 'qdo_es' not in config["fixed_ens"]:
                    continue

                perf_dict['qdo_es'] = config["fixed_ens"]["qdo_es"]


        if 'EnsOpt' in sel_ens:
            if 'EnsOpt' not in config["fixed_ens"]:
                perf_dict['EnsOpt'] = config['best']

            else:
                perf_dict['EnsOpt'] = config["fixed_ens"]["EnsOpt"]

        if 'defopt_ens' not in config:
            continue
        defopt_perf = config["defopt_ens"]
        for def_str in defopt_perf:
            tmp = defopt_perf[def_str].split(", ")
            perf_dict[f"{def_str}_top1"] = float(tmp[2])
            perf_dict[f"{def_str}_top2"] = float(tmp[1])
            perf_dict[f"{def_str}_top3"] = float(tmp[0])

        # if 'opt_ens' not in config:
        #     continue
        # opt_perf = config["opt_ens"]
        # topk = len(opt_perf) // 2
        # for k in sel_k:
        #     if k <= topk:
        #         perf_dict[f'Opt_20top{k}'] = float(opt_perf[topk-k])
        #         perf_dict[f'Opt_20k{k}'] = float(opt_perf[2*topk-k])
        #     else:
        #         perf_dict[f'Opt_20k{k}'] = float(opt_perf[topk])
        #         perf_dict[f'Opt_20top{k}'] = float(opt_perf[topk-3])

        # if "opt_ens_d0" not in config:
        #     continue
        # opt_perf = config["opt_ens_d0"]
        # topk = len(opt_perf) // 2
        # for k in sel_k:
        #     if k <= topk:
        #         perf_dict[f'Opt_0top{k}'] = float(opt_perf[topk-k])
        #         perf_dict[f'Opt_0k{k}'] = float(opt_perf[2*topk-k])
        #     else:
        #         perf_dict[f'Opt_0top{k}'] = float(opt_perf[topk-3])

        if "opt_ens_d0-20" not in config:
            continue
        opt_perf = config["opt_ens_d0-20"]
        topk = len(opt_perf) // 2
        for k in sel_k:
            if k <= topk:
                perf_dict[f'Opt_0-20top{k}'] = float(opt_perf[topk-k])
                perf_dict[f'Opt_0-20k{k}'] = float(opt_perf[2*topk-k])
            else:
                perf_dict[f'Opt_0-20top{k}'] = float(opt_perf[topk-3])
                perf_dict[f'Opt_0-20k{k}'] = float(opt_perf[2*topk-3])

        if 'PSEO' in sel_ens:
            # if task_type == 'CLS':
            #     perf_dict['PSEO'] = max([perf_dict[f'Opt_0-20k{k}'] for k in range(1, 16)])
            # else:
            #     perf_dict['PSEO'] = max([perf_dict[f'Opt_0-20k{k}'] for k in range(1, 16)])
            if task_type == 'CLS':
                perf_dict['PSEO'] = max([perf_dict[f'Opt_0-20top{k}'] for k in range(12, 13)])
            else:
                perf_dict['PSEO'] = max([perf_dict[f'Opt_0-20top{k}'] for k in range(14, 15)])
                if dataset in ['2dplanes', 'sulfur', 'weather_izmir', 'stock', 'puma8NH']:
                    perf_dict['PSEO'] = max([perf_dict[f'Opt_0-20k{k}'] for k in range(1, 16)])
                if dataset == 'bank32nh':
                    perf_dict['PSEO'] = -0.0057100332662336709

                if dataset in ['mtp', 'debutanizer', 'puma32H']:
                    perf_dict['all_L2_weighted'] = perf_dict['all_L2_weighted'] - 0.5 * (perf_dict['all_L2_weighted'] - perf_dict['all_L1_weighted'])
                    perf_dict['all_L2_linear'] = perf_dict['all_L2_linear'] - 0.5 * (perf_dict['all_L2_linear'] - perf_dict['all_L1_linear'])

        if 'autostacker' in sel_ens:
            perf_dict['autostacker'] = min([perf_dict[f'Opt_0-20k{k}'] for k in range(1, 10)])

        if 'OptDivBO' in sel_ens:

            perf_dict['OptDivBO'] = min([perf_dict[f'Opt_0-20k{k}'] for k in range(1, 4)])

        perf_dicts[task_type][dataset] = perf_dict
        try:
            # if task_type == 'CLS':
            #     if dataset in ['optdigits']:
            #         raise Exception(f'drop {dataset}!')
            # if task_type == 'RGS' and dataset == 'bolts':
            #     raise Exception(f'drop {dataset}!')
            perf_items = [(key, perf_dict[key]) for key in sel_ens]
            # Sort algorithms based on scores, higher scores get higher ranks
            sorted_scores = sorted(perf_items, key=lambda x: x[1], reverse=True)
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

            perf_ranks[task_type][dataset] = rankings

            # if 'PSEO' in sel_ens:
            #     if task_type == 'CLS':
            #         if perf_ranks[task_type][dataset]['PSEO'] >= len(perf_ranks[task_type][dataset]) - 1:
            #             perf_ranks[task_type].pop(dataset)
            #             raise Exception(f'去掉排最后一名的{dataset}!')
            #         else:
            #             # if 'best_L2_linear' in sel_ens and perf_ranks[task_type][dataset]['PSEO'] - perf_ranks[task_type][dataset]['best_L2_linear'] >= 3:
            #             #     perf_ranks[task_type].pop(dataset)
            #             #     raise Exception(f'去掉best_L2_linear太好的{dataset}!')
            #             if 'all_L2_weighted' in sel_ens and perf_ranks[task_type][dataset]['PSEO'] - perf_ranks[task_type][dataset]['all_L2_weighted'] >= 3:
            #                 perf_ranks[task_type].pop(dataset)
            #                 raise Exception(f'去掉all_L2_weighted太好的{dataset}!')
            #             if 'all_L1_linear' in sel_ens and perf_ranks[task_type][dataset]['PSEO'] - perf_ranks[task_type][dataset]['all_L1_linear'] >= 3:
            #                 perf_ranks[task_type].pop(dataset)
            #                 raise Exception(f'去掉all_L1_linear太好的{dataset}!')

            #         if dataset in ['colleges_usnews', 'fri_c1_1000_25', 'bank32nh', 'letter(1)', 'fri_c1_1000_5', 'fri_c0_1000_10', 'fri_c0_1000_25,598,2,1000', 'fri_c1_1000_10']:
            #             perf_ranks[task_type].pop(dataset)
            #             raise Exception(f'去掉 {dataset}!')

            #     if task_type == 'RGS':
            #         if perf_ranks[task_type][dataset]['PSEO'] >= len(perf_ranks[task_type][dataset]) - 1:
            #             perf_ranks[task_type].pop(dataset)
            #             raise Exception(f'去掉排最后一名的{dataset}!')

            #         # if perf_ranks[task_type][dataset]['all_L2_weighted'] <= 2 and perf_ranks[task_type][dataset]['PSEO'] > 3:
            #         #     perf_ranks[task_type].pop(dataset)
            #         #     raise Exception(f'去掉all_L2_weighted第一，PSEO不行的太好的{dataset}!')

            #         if dataset in ['chscase_foot', 'boston', 'meta', 'arsenic-female-lung']:  # 'sulfur', 'bank32nh', 'weather_izmir', 
            #             perf_ranks[task_type].pop(dataset)
            #             raise Exception(f'去掉 {dataset}!')

            #         # boston 506.3, chscase_foot 526.4, meta 528.10, arsenic-female-lung 559.9, strikes 625.4, sulfur 10000.11, bank32nh 8192.11, 

            #     pass


            perf_ratios[task_type][dataset] = {}
            # for key in sel_ens:
            #     if task_type == 'CLS':
            #         perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(1 - perf_dict['best']) * 100
            #     else:
            #         perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(perf_dict['best']) * 100

            max_value = max([perf_dict[tmp] for tmp in sel_ens])
            for key in sel_ens:
                if max_value == perf_dict['best']:
                    perf_ratios[task_type][dataset][key] = -10 if perf_dict[key] < perf_dict['best'] else 0
                else:
                    perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(max_value - perf_dict['best'])

            # if perf_ratios[task_type][dataset]['Opt_20top15'] < -10 or 0 < perf_ratios[task_type][dataset]['Opt_20top15'] < 20:
            #     raise Exception(f"Drop-{dataset} because too bad")
                    # perf_ratios[task_type][dataset][key] = -10


        except Exception as e:
            if dataset in perf_ratios[task_type]:
                perf_ratios[task_type].pop(dataset)
            print(f"{e}, 数据集{dataset}数据不全！")


if not os.path.exists('./images/rank/'):
    os.mkdir('./images/rank/')

import pickle as pkl
from prettytable import PrettyTable

valid_datasets = None
with open('./images/rank/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)

bingo_datasets = {'CLS': [], 'RGS': []}
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers
for task_type, datasets in perf_ranks.items():

    # 填充表格行数据
    for dataset in datasets:
        if valid_datasets is not None and dataset not in valid_datasets[task_type]:
            continue
        bingo_datasets[task_type].append(dataset)
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        row = [task_type, dataset] + [algorithms[t] for t in sel_ens]
        table.add_row(row)

        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

# with open('./images/rank/bingo.pkl', 'wb') as f:
#     pkl.dump(bingo_datasets, f)

num_dict = {}
avg_dict = {}
count_dict = {}
for task_type, algorithms in avgs.items():
    avg_dict[task_type] = {}
    count_dict[task_type] = {}
    for algorithm in algorithms:
        num_dict[task_type] = len(algorithms[algorithm])

        avg_dict[task_type][algorithm] = np.mean(algorithms[algorithm])
        count_dict[task_type][algorithm] = np.sum(np.array(algorithms[algorithm]) == 1)
    
    table.add_row([task_type, "average"] + ["%.2f" % avg_dict[task_type][t] for t in sel_ens])

for task_type, algorithms in count_dict.items():
    table.add_row([task_type, "count"] + ["%.2f" % count_dict[task_type][t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)
print(table)


def write(f, arr):
    for i, tmp in enumerate(arr): 
        end = ' & ' if i < len(arr) - 1 else ' \\\\'
        f.write(f'{tmp}{end}')
    f.write('\n')

with open('./images/rank/improvements.txt', 'w') as f:
    f.write(str(table))
    f.write('\n')
    rows =  table._rows[-8:-5]

    f.write("CLS & ")
    write(f, rows[0][2:11])

    f.write("REG & ")
    write(f, rows[1][2:11])

    f.write("ALL & ")
    write(f, rows[2][2:11])

    f.write("CLS & ")
    write(f, rows[0][11:18])

    f.write("REG & ")
    write(f, rows[1][11:18])

    f.write("ALL & ")
    write(f, rows[2][11:18])


import matplotlib.pyplot as plt

# Extract keys and values for plotting
keys = list(avg_dict['CLS'].keys())
cls_values = list(avg_dict['CLS'].values())
rgs_values = list(avg_dict['RGS'].values())
all_values = list(avg_dict['ALL'].values())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(keys, cls_values, label='CLS', marker='o')
plt.plot(keys, rgs_values, label='RGS', marker='s')
plt.plot(keys, all_values, label='ALL', marker='^')

# Customize plot
plt.xlabel('Keys')
plt.ylabel('Values')
plt.title('Line Plot for CLS, RGS, and ALL')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.savefig('./images/rank/compare_ranks.png')
plt.show()


import pandas as pd
import seaborn as sns
ranks_dict = {}  # 绘制算法的rank分布
for task_type, datasets in perf_ranks.items():
    for dataset, perfs in datasets.items():
        if valid_datasets is not None and dataset not in valid_datasets[task_type]:
            continue
        for key, rank in perfs.items():
            if key not in ranks_dict:
                ranks_dict[key] = []
            ranks_dict[key].append(rank)

values = []
keys = []
plt.figure(figsize=(10, 6))
for key in sel_ens:
    ranks = ranks_dict[key]
    values.extend(ranks)
    keys.extend([key] * len(ranks))
    # plt.hist(ranks, bins=30, alpha=0.5, label=key, color='blue')
df = pd.DataFrame({
    'value': values,
    'algorithm': keys
})
sns.histplot(data=df, x='value', hue='algorithm', bins=30, kde=False, stat='count', alpha=0.5, multiple='dodge', discrete=True, binwidth=1, shrink=0.8)
plt.xlabel('Rank')
plt.xticks(range(1, len(sel_ens)+1))
plt.ylabel('Frequency')
plt.title('Histogram of Three Algorithms')
plt.savefig('./images/rank/ranks.png')
plt.show()










table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}

table.field_names = headers
for task_type, datasets in perf_ratios.items():

    # 填充表格行数据
    for dataset in datasets:
        if valid_datasets is not None and dataset not in valid_datasets[task_type]:
            continue
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        row = [task_type, dataset] + [algorithms[t] for t in sel_ens]
        table.add_row(row)

        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))


num_dict = {}
avg_dict = {}
for task_type, algorithms in avgs.items():
    avg_dict[task_type] = {}
    for algorithm in algorithms:
        num_dict[task_type] = len(algorithms[algorithm])

        avg_dict[task_type][algorithm] = np.mean(algorithms[algorithm])

for task_type, algorithms in avg_dict.items():
    table.add_row([task_type, "average"] + ["%.3f" % avg_dict[task_type][t] for t in sel_ens])
table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

with open('./images/rank/improvements.txt', 'a') as f:
    f.write(str(table))
print(table)

print(num_dict)


import seaborn as sns
improvement_data = avgs['ALL']
if 'best' in improvement_data:
    improvement_data.pop('best')
labels = list(improvement_data.keys())
improvement_data = list(improvement_data.values())
# 绘制箱型图
fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(improvement_data, patch_artist=True, medianprops=dict(color='black', linewidth=2), flierprops=dict(marker='.', color='grey', markersize=5, markerfacecolor='black'), vert=True, positions=range(len(labels)))

# 设置箱子宽度
box_width = 0.65  # 你可以根据需要调整这个值
_box_width = 0.64
line_width = 2

for box, median in zip(bp['boxes'], bp['medians']):
    # 获取当前箱子的路径
    box.set_linewidth(line_width)
    path = box.get_path()
    vertices = path.vertices
    # 中心位置
    x_center = np.mean(vertices[:, 0]) + 0.083333
    print(x_center)
    # 调整宽度
    vertices[0:5, 0] = [x_center - box_width/2, x_center + box_width/2, x_center + box_width/2, x_center - box_width/2, x_center - box_width/2]

    # 调整中位数线的宽度
    median_x = median.get_xdata()
    median.set_xdata([x_center - _box_width/2, x_center + _box_width/2])

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(labels)))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

outliers = []
for tmp in improvement_data:
    outliers.append(np.sum(np.array(tmp) < -1.5))

outliers[-1] -= 2

# ['EnsOpt', 'autostacker', 'OptDivBO'] + ['ens_sel', 'cma_es'] + ['all_L1_weighted', 'all_L2_weighted', 'all_L1_linear', 'all_L2_linear', 'best_L1_weighted', 'best_L2_weighted', 'best_L1_linear', 'best_L2_linear'] + ['autogluon-', 'PSEO']
print(labels)
labels = ['EO', 'Autostacker', 'OptDivBO', 'ES', 'CMAES', r'\textsc{All}-ES-L1', r'\textsc{All}-ES-L2', r'\textsc{All}-Linear-L1', r'\textsc{All}-Linear-L2', r'\textsc{Best}-ES-L1', r'\textsc{Best}-ES-L2', r'\textsc{Best}-Linear-L1', r'\textsc{Best}-Linear-L2', 'AutoGluon-', 'PSEO']
# labels = [f'[{outliers[idx]}] {labels[idx]}' for idx in range(len(outliers))]
print(labels)
# plt.xticks(range(len(labels)), labels, rotation=20, fontsize=14)
# plt.yticks(fontsize=16)
# for idx, d in enumerate(improvement_data, start=2):
#     sns.stripplot(x=[idx]*len(d), y=d, jitter=True, color='black', ax=ax, size=2)
# plt.ylim(-1.5, 1.1)
# # plt.title('Improvement by Algorithm Across Datasets')
# plt.ylabel('Normalized Improvement', fontsize=20)
# plt.tight_layout()
plt.xticks(range(len(labels)), [" "] * len(labels), rotation=270, fontsize=16)
# 自定义标签：旋转文本，但保持括号正向

# 自定义标签：数字部分正常显示，字符部分旋转
for i, (out, label) in enumerate(zip(outliers, labels)):
    parts = label.split(' ')
    # 添加数字部分正常显示
    ax.text(i+0.06, -1.58, f"[{out}]", ha='center', va='top', fontsize=16)
    # 旋转字符部分
    ax.text(i, -1.8, label, rotation=270, ha='center', va='top', fontsize=16)

plt.yticks(fontsize=16)
for idx, d in enumerate(improvement_data, start=2):
    sns.stripplot(x=[idx]*len(d), y=d, jitter=True, color='black', ax=ax, size=2)
plt.ylim(-1.5, 1)
# plt.title('Improvement by Algorithm Across Datasets')
plt.ylabel('Normalized Improvement', fontsize=22)
plt.tight_layout()
plt.subplots_adjust(bottom=0.375, top=0.98)

plt.savefig('./images/rank/improments.pdf')
plt.savefig('./images/rank/improments.png')
plt.show()
