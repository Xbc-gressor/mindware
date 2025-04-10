import os
import json
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info

data_dir = './block012_data'
data_dir = './compress_data'
data_dir = './ens_stackskip_data'

# times_dict = {'CLS': {}, 'RGS': {}}
# num_dict = {'CLS': {}, 'RGS': {}}

# for sub_dir in sorted(os.listdir(data_dir)):

#     if not os.path.exists(os.path.join(data_dir, sub_dir, 'config.json')):
#         continue

#     with open(os.path.join(data_dir, sub_dir, 'config.json'), 'r') as f:
#         config = json.load(f)

#     if not os.path.exists(os.path.join(data_dir, sub_dir, 'best_model_info.json')):
#         continue

#     with open(os.path.join(data_dir, sub_dir, 'best_model_info.json'), 'r') as f:
#         best_model_info = json.load(f)

#     types = config['task_type']
#     task_id = config['task_id']
#     folds = best_model_info["ensemble"]["folds"]
#     opt = f"{folds[0]}_{folds[1]}"
#     layer_losses = best_model_info["ensemble"]["layer_loss"]

#     if types == 4:
#         tar_dict = times_dict['RGS']
#         tar_dict1 = num_dict['RGS']
#     else:
#         tar_dict = times_dict['CLS']
#         tar_dict1 = num_dict['CLS']

#     if task_id not in tar_dict:
#         tar_dict[task_id] = {}
#     if task_id not in tar_dict1:
#         tar_dict1[task_id] = {}

#     _max = np.max([layer_losses[0], layer_losses[1]], axis=0)
#     tar_dict[task_id][opt] = round(np.mean(_max), 5)
#     tar_dict1[task_id][opt] = np.sum(np.array(layer_losses[1]) > np.array(layer_losses[0]))

# rank_fields = list(list(list(times_dict.values())[0].values())[0].keys())

# from prettytable import PrettyTable

# for tar_dict in [times_dict, num_dict]:
#     table = PrettyTable()
#     headers = ["Task Type", "Dataset"] + rank_fields
#     avgs = {
#         "CLS": {t:[] for t in headers[2:]},
#         "RGS": {t:[] for t in headers[2:]},
#         "ALL": {t:[] for t in headers[2:]}
#     }

#     for task_type, datasets in tar_dict.items():
        
#         table.field_names = headers
        
#         dataset_names = None
#         if task_type == "CLS":
#             dataset_names = [n for n in cls_info.index if n in datasets]
#         else:
#             dataset_names = [n for n in rgs_info.index if n in datasets]
        
#         # 填充表格行数据
#         for dataset in dataset_names:
#             algorithms = datasets[dataset]
#             if algorithms == {}:
#                 continue
#             row = [task_type, dataset] + [algorithms.get(t, None) for t in headers[2:]]
#             table.add_row(row)
            
#             for t in algorithms:
#                 avgs[task_type][t].append(algorithms[t])
#                 avgs["ALL"][t].append(algorithms[t])
#         table.add_row(["-"*9, "-"*12] + ["-"*11] * len(headers[2:]))
            

#     print(table)
# breakpoint()

# 看一下选择情况

times_dict = {'CLS': {}, 'RGS': {}}

for sub_dir in sorted(os.listdir(data_dir)):

    if not os.path.exists(os.path.join(data_dir, sub_dir, 'config.json')):
        continue

    with open(os.path.join(data_dir, sub_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    if not os.path.exists(os.path.join(data_dir, sub_dir, 'best_model_info.json')):
        continue

    with open(os.path.join(data_dir, sub_dir, 'best_model_info.json'), 'r') as f:
        best_model_info = json.load(f)

    types = config['task_type']
    task_id = config['task_id']
    if 'folds' not in best_model_info["ensemble"]:
        continue
    opt = f"{best_model_info['ensemble']['folds'][0]}_{best_model_info['ensemble']['folds'][1]}"
    layer = best_model_info["ensemble"]["stask_layers"]
    method = best_model_info["ensemble"]["meta_learner"]
    ens = f"{method}_L{layer+1}"

    if types == 4:
        tar_dict = times_dict['RGS']
    else:
        tar_dict = times_dict['CLS']

    if task_id not in tar_dict:
        tar_dict[task_id] = {}

    tar_dict[task_id][opt] = ens

rank_fields = list(list(list(times_dict.values())[0].values())[0].keys())

from prettytable import PrettyTable

for tar_dict in [times_dict]:
    table = PrettyTable()
    headers = ["Task Type", "Dataset"] + rank_fields
    avgs = {
        "CLS": {t:[] for t in headers[2:]},
        "RGS": {t:[] for t in headers[2:]},
        "ALL": {t:[] for t in headers[2:]}
    }

    for task_type, datasets in tar_dict.items():
        
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
            row = [task_type, dataset] + [algorithms.get(t, None) for t in headers[2:]]
            table.add_row(row)
            
            for t in algorithms:
                avgs[task_type][t].append(algorithms[t])
                avgs["ALL"][t].append(algorithms[t])
        table.add_row(["-"*9, "-"*12] + ["-"*11] * len(headers[2:]))
            

    print(table)
breakpoint()
