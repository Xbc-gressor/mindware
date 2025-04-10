import os
import json
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info

data_dir = './block012_data'
data_dir = './compress_data'
data_dir = './ens_para_stackskip_data'

times_dict = {'CLS': {}, 'RGS': {}}

for sub_dir in sorted(os.listdir(data_dir)):
    
    config_path = os.path.join(data_dir, sub_dir, 'config.json')
    best_model_info_path = os.path.join(data_dir, sub_dir, 'best_model_info.json')
    if not (os.path.exists(config_path) and os.path.exists(best_model_info_path)):
        continue

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config['ensemble_method'] is None:
        continue
        
    with open(best_model_info_path, 'r') as f:
        best_model_info = json.load(f)
    
    thread = best_model_info['ensemble']['thread']
    types = config['task_type']
    task_id = config['task_id']
    times = np.mean(best_model_info['ensemble']['train_cost'])

    if types == 4:
        tar_dict = times_dict['RGS']
    else:
        tar_dict = times_dict['CLS']

    if task_id not in tar_dict:
        tar_dict[task_id] = {}

    opt = best_model_info['ensemble']["ensemble_method"] + f"_thr{thread}"
        
    tar_dict[task_id][opt] = times
    
    
rank_fields = list(list(list(times_dict.values())[0].values())[0].keys())
idx = np.argsort([int(tmp.split('thr')[1]) for tmp in rank_fields])
rank_fields = [rank_fields[tmp] for tmp in idx]
    
from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + rank_fields
avgs = {
    "CLS": {t:[] for t in headers[2:]},
    "RGS": {t:[] for t in headers[2:]},
    "ALL": {t:[] for t in headers[2:]}
}

for task_type, datasets in times_dict.items():
    
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
        row = [task_type, dataset] + ["%.3f" % algorithms[t] for t in headers[2:]]
        table.add_row(row)
        
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(headers[2:]))

for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        algorithms[algorithm] = np.mean(algorithms[algorithm])
    
    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in headers[2:]])
        

print(table)

import matplotlib.pyplot as plt
# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 绘制每个子图
for ax, (key, values) in zip(axes, avgs.items()):
    x = np.array([int(k.split('thr')[1]) for k in values.keys()])
    y = np.array(list(values.values()))
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    ax.plot(x, y, marker='o')
    ax.set_title(key)
    ax.set_xticks(x)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time of Stacking-size10-L2')

# 调整布局
plt.tight_layout()
plt.savefig('./images/multipro.png')
plt.show()

breakpoint()

