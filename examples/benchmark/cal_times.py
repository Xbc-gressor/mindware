import os
import json
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info

data_dir = './block012_data'
data_dir = './compress_data'

times_dict = {'CLS': {}, 'RGS': {}}

for sub_dir in sorted(os.listdir(data_dir)):
    
    with open(os.path.join(data_dir, sub_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        
    with open(os.path.join(data_dir, sub_dir, 'best_model_info.json'), 'r') as f:
        best_model_info = json.load(f)
        
    opt = config['optimizer']
    if opt != 'block_1':
        continue
    
    types = config['task_type']
    task_id = config['task_id']
    times = len(best_model_info['opt_trajectory']['final_rewards'])
    
    if types == 4:
        tar_dict = times_dict['RGS']
    else:
        tar_dict = times_dict['CLS']
    
    if task_id not in tar_dict:
        tar_dict[task_id] = {}

    if 'include_preprocessors' in config:
        tmp = config['include_preprocessors']
        if tmp[list(tmp.keys())[0]] != None:
            opt = "filall_" + opt
            
    if ('film_' + opt) not in tar_dict[task_id]:
        opt = 'film_' + opt
        
    tar_dict[task_id][opt] = times
    
    
rank_fields = list(list(list(times_dict.values())[0].values())[0].keys())
    
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
        row = [task_type, dataset] + [algorithms[t] for t in headers[2:]]
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
breakpoint()

