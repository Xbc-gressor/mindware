import os
import json
import numpy as np


times_dict = {'CLS': {}, 'RGS': {}}


for task_type in ['CLS', 'RGS']:
    tt = 3600 if task_type == 'CLS' else 7200
    sub_dir = os.path.join('./benchmark_data/', f'data_{task_type}_{tt}')
    for sub2_dir in os.listdir(sub_dir):
        sub2_dir = os.path.join(sub_dir, sub2_dir)

        valid_paths = []
        task_id = None
        for sub3_dir in sorted(os.listdir(sub2_dir)):
            if sub3_dir.startswith('CASHFE'):

                sub3_dir = os.path.join(sub2_dir, sub3_dir)
                config_path = os.path.join(sub3_dir, './config.json')
                best_path = os.path.join(sub3_dir, './best_model_info.json')
                if not os.path.exists(config_path) or not os.path.exists(best_path):
                    continue

                with open(config_path, 'r') as f:
                    config = json.load(f)

                with open(best_path, 'r') as f:
                    best_config = json.load(f)

                task_id = config['task_id']
                times_dict[task_type][task_id] = {
                    'bo': len(best_config["opt_trajectory"]["action_sequence"])
                }
            elif sub3_dir.startswith('ENS-'):
                sub3_dir = os.path.join(sub2_dir, sub3_dir)
                config_path = os.path.join(sub3_dir, './config.json')
                best_path = os.path.join(sub3_dir, './best_model_info.json')
                if not os.path.exists(config_path) or not os.path.exists(best_path):
                    continue

                with open(config_path, 'r') as f:
                    config = json.load(f)

                if config['time_limit'] == 1:
                    continue

                with open(best_path, 'r') as f:
                    best_config = json.load(f)

                valid_paths.append((sub3_dir, best_config["comb_count"]))

                task_id = config['task_id']

        if task_id not in times_dict[task_type]:
            continue
        if len(valid_paths) > 0:
            times_dict[task_type][task_id].update({
                '0-20': valid_paths[-1][1],
            })
        else:
            times_dict[task_type].pop(task_id)
    
    
rank_fields = list(list(list(times_dict.values())[0].values())[0].keys())
    
from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + rank_fields
avgs = {
    "CLS": {t:[] for t in headers[2:]},
    "RGS": {t:[] for t in headers[2:]},
    "ALL": {t:[] for t in headers[2:]}
}

import pickle as pkl
with open('./images/rank/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)
sel_ens = ["bo", '0-20']  #, '20', '0', '0-20'
table.field_names = headers
for task_type, datasets in times_dict.items():
    # 填充表格行数据
    for dataset in datasets:
        algorithms = datasets[dataset]
        if algorithms == {}:
            continue
        if dataset not in valid_datasets[task_type]:
            continue
        row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
        table.add_row(row)
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

num_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        num_dict[task_type] = len(algorithms[algorithm])
        algorithms[algorithm] = np.mean(algorithms[algorithm])

    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in headers[2:]])
table.add_row(headers)


print(table)
print(num_dict)
