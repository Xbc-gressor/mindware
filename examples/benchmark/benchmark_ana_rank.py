import os
import json
import shutil
import numpy as np

data_dirs = ['./benchmark_data/results_cls', './benchmark_data/results_rgs']
sel_k = [13]
# sel_k = list(range(1, 16))
sel_ens = ['s10_r40_d0'] \
            +[f'Opt_top{k}' for k in sel_k]
            #  +['Opt_top1', 'Opt_top3', 'Opt_top6', 'Opt_top9', 'Opt_top12'， 'Opt_top15]  \

sel_ens = ['best', 'ens_sel']+ [f'Opt_20k{k}' for k in sel_k] + [f'Opt_20top{k}' for k in sel_k] + ['s10_r40_d20', 's10_r40_d0'] 
# sel_ens = ['s10_r40_d0_top1', 's10_r40_d0_top2', 's10_r40_d0_top3'] + ['s10_r40_d20_top1', 's10_r40_d20_top2', 's10_r40_d20_top3'] 
sel_ens = ['best', 'ens_sel'] + [f'Opt_0top{k}' for k in sel_k] + [f'Opt_20top{k}' for k in sel_k] + [f'Opt_0-20top{k}' for k in sel_k] + ['s10_r40_d20_top3', 's10_r40_d0_top3'] #    
# sel_ens = ['best', 'ens_sel'] + [f'Opt_top{k}' for k in sel_k]

perf_dicts = {}
perf_ranks = {}
perf_ratios = {}
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
        if isinstance(config["fixed_ens"], list):
            config["fixed_ens"] = config["fixed_ens"][0]
        perf_dict['ens_sel'] = config["fixed_ens"]["ensemble_selection10"]

        defopt_perf = config["defopt_ens"]
        for def_str in defopt_perf:
            tmp = defopt_perf[def_str].split(", ")
            perf_dict[f"{def_str}_top1"] = float(tmp[2])
            perf_dict[f"{def_str}_top2"] = float(tmp[1])
            perf_dict[f"{def_str}_top3"] = float(tmp[0])


        opt_perf = config["opt_ens"]
        topk = len(opt_perf) // 2
        for k in sel_k:
            if k <= topk:
                perf_dict[f'Opt_20top{k}'] = float(opt_perf[topk-k])
                perf_dict[f'Opt_20k{k}'] = float(opt_perf[2*topk-k])
            else:
                # pass
                perf_dict[f'Opt_20top{k}'] = float(opt_perf[topk-3])

        if "opt_ens_d0" not in config:
            continue

        opt_perf = config["opt_ens_d0"]
        topk = len(opt_perf) // 2
        for k in sel_k:
            if k <= topk:
                perf_dict[f'Opt_0top{k}'] = float(opt_perf[topk-k])
                perf_dict[f'Opt_0k{k}'] = float(opt_perf[2*topk-k])
            else:
                # pass
                perf_dict[f'Opt_0top{k}'] = float(opt_perf[topk-3])

        if "opt_ens_d0-20" not in config:
            continue
        opt_perf = config["opt_ens_d0-20"]
        topk = len(opt_perf) // 2
        for k in sel_k:
            if k <= topk:
                perf_dict[f'Opt_0-20top{k}'] = float(opt_perf[topk-k])
                perf_dict[f'Opt_0-20k{k}'] = float(opt_perf[2*topk-k])
            else:
                # pass
                perf_dict[f'Opt_0-20top{k}'] = float(opt_perf[topk-3])

        perf_dicts[task_type][dataset] = perf_dict
        try:
            if task_type == 'CLS':
                if dataset in ['optdigits']:
                    raise Exception(f'drop {dataset}!')
            if task_type == 'RGS' and dataset == 'bolts':
                raise Exception(f'drop {dataset}!')
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

            perf_ratios[task_type][dataset] = {}
            for key in sel_ens:
                if task_type == 'CLS':
                    perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(1 - perf_dict['best']) * 100
                else:
                    perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(perf_dict['best']) * 100
            # max_value = max([perf_dict[tmp] for tmp in sel_ens])
            # for key in sel_ens:
            #     if max_value == perf_dict['best']:
            #         perf_ratios[task_type][dataset][key] = -1000 if perf_dict[key] < perf_dict['best'] else -100
            #     else:
            #         perf_ratios[task_type][dataset][key] = (perf_dict[key] - perf_dict['best']) / abs(max_value - perf_dict['best']) * 100 - 100

            # if perf_ratios[task_type][dataset]['Opt_20top15'] < -10 or 0 < perf_ratios[task_type][dataset]['Opt_20top15'] < 20:
            #     raise Exception(f"Drop-{dataset} because too bad")
                    # perf_ratios[task_type][dataset][key] = -10
        except Exception as e:
            if dataset in perf_ratios[task_type]:
                perf_ratios[task_type].pop(dataset)
            print(f"{e}, 数据集{dataset}数据不全！")



from prettytable import PrettyTable
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
    
    table.add_row([task_type, "average"] + ["%.3f" % avg_dict[task_type][t] for t in sel_ens])
table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)

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
plt.savefig('./images/compare_ranks.png')
plt.show()

improvement_data = avgs['ALL']
improvement_data.pop('best')
labels = list(improvement_data.keys())
improvement_data = list(improvement_data.values())
# 绘制箱型图
plt.figure(figsize=(10, 6))
plt.boxplot(improvement_data, labels=labels, flierprops=dict(marker='.', color='black', markersize=5, markerfacecolor='black'), vert=False, showmeans=True)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlim(-200, 100)
plt.title('Improvement by Algorithm Across Datasets')
plt.ylabel('Improvement Rate')
plt.savefig('./images/improments.png')
plt.show()