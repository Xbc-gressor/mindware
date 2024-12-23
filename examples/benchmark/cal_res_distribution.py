import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
import json
from collections import Counter
from mindware.components.utils.constants import *
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info


import numpy as np

def parse_res(file_dir):
    results = {
        "CLS": {},
        "RGS": {},
    }
    
    for file in os.listdir(file_dir):
        base_dir = os.path.join(file_dir, file)
        with open(os.path.join(base_dir, "config.json"), 'r') as file:
            opt_config = json.load(file)
        
        name = opt_config["name"]
        task_type = opt_config["task_type"]
        dataset = opt_config["task_id"]
        
        tar_res = results["RGS"]
        if task_type in CLS_TASKS:
            tar_res =results["CLS"]
        if dataset not in tar_res:
            tar_res[dataset] = {}
            
        
        with open(os.path.join(base_dir, "best_model_info.json"), 'r') as file:
            best_config = json.load(file)
        if name == "cashfe":
            conf = best_config["best"][1]
            tar_res[dataset]["cashfe_model"] = conf["algorithm"]
            bal = conf["balancer"] if "balancer" in conf else "\\"
            tar_res[dataset]["cashfe_balancer"] = bal
            tar_res[dataset]["cashfe_preprocessor"] = conf["preprocessor"]
            tar_res[dataset]["cashfe_rescaler"] = conf["rescaler"]
        elif name == "cash":
            conf = best_config["best"][1]
            tar_res[dataset]["cash_model"] = conf["algorithm"]
            
    return results

file_dir = "/root/mindware/examples/benchmark/cash_vs_cashfe_data"
results = parse_res(file_dir)

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset", "cash_model", "cashfe_model", "cashfe_balancer", "cashfe_preprocessor", "cashfe_rescaler"]
avgs = {
    "CLS": {t:[] for t in headers[2:]},
    "RGS": {t:[] for t in headers[2:]},
    "ALL": {t:[] for t in headers[2:]}
}
for task_type, datasets in results.items():
    
    table.field_names = headers
    
    dataset_names = None
    if task_type == "CLS":
        dataset_names = [n for n in cls_info.index if n in datasets]
    else:
        dataset_names = [n for n in rgs_info.index if n in datasets]
        
    
    # 填充表格行数据
    for dataset in dataset_names:
        algorithms = datasets[dataset]
        row = [task_type, dataset] + [algorithms[t] for t in headers[2:]]
        table.add_row(row)
        
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12, "-"*17, "-"*17, "-"*18, "-"*32,  "-"*25])

for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        # 使用Counter来计数列表中各元素的出现次数
        counter = Counter(algorithms[algorithm])
        # most_common(1)返回一个列表，其中包含一个元组（出现次数最多的元素及其计数）
        most_common_element, count = counter.most_common(1)[0]
        algorithms[algorithm] = f"{most_common_element}({count})"
    
    table.add_row([task_type, "most common"] + [algorithms[t] for t in headers[2:]])
        

print(table)
