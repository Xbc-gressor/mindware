import re
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info

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
            match = re.match(r"(CLS|RGS): (cashfe|cash), (\w+): (-?\d+\.\d+), (-?\d+\.\d+)", line)
            if match:
                task_type, algorithm, dataset, best, ens = match.groups()
                if dataset not in results[task_type]:
                    results[task_type][dataset] = {}
                results[task_type][dataset][f"{algorithm}_best"] = float(best)
                results[task_type][dataset][f"{algorithm}_ens"] = float(ens) 

    ranks = {
        "CLS": {},
        "RGS": {},
    }
    for task_type in results:
        for dataset in results[task_type]:
            if dataset not in ranks[task_type]:
                ranks[task_type][dataset] = {}
                
            datas = results[task_type][dataset]
            # Sort algorithms based on scores, higher scores get higher ranks
            sorted_scores = sorted(datas.items(), key=lambda x: x[1], reverse=True)
            # Assign ranks
            ranks[task_type][dataset] = {alg: rank + 1 for rank, (alg, score) in enumerate(sorted_scores)}

    return results, ranks

def calculate_averages(results):
    for task_type in results:
        for algorithm in results[task_type]:
            avg_best = sum(results[task_type][algorithm]["best"]) / len(results[task_type][algorithm]["best"])
            avg_ens = sum(results[task_type][algorithm]["ens"]) / len(results[task_type][algorithm]["ens"])
            print(f"{task_type} {algorithm} average best: {avg_best}")
            print(f"{task_type} {algorithm} average ens: {avg_ens}")

# Replace 'path_to_your_file.txt' with the actual path to your data file
file_path = '/root/mindware/examples/benchmark/results.txt'
results, ranks = parse_data(file_path)

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset", "cashfe_best", "cashfe_ens", "cash_best", "cash_ens"]
avgs = {
    "CLS": {t:[] for t in headers[2:]},
    "RGS": {t:[] for t in headers[2:]},
    "ALL": {t:[] for t in headers[2:]}
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
        row = [task_type, dataset] + [algorithms[t] for t in headers[2:]]
        table.add_row(row)
        
        for t in algorithms:
            avgs[task_type][t].append(algorithms[t])
            avgs["ALL"][t].append(algorithms[t])
    table.add_row(["-"*9, "-"*12, "-"*11, "-"*11, "-"*11, "-"*11])

for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        algorithms[algorithm] = np.mean(algorithms[algorithm])
    
    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in headers[2:]])
        

print(table)
