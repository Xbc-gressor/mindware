import re
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info


# rank_fields = ['cashfe_best', 'cashfe_ens', 'cashfe-block_1_best', 'cashfe-block_1_ens']  # 比较mab和block_1
rank_fields = ['cashfe-block_1_ens', 'autogluon_ens']  # 比较mab和block_1

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
            match = re.match(r"(CLS|RGS): (cashfe|cash|cashfe-block_0|cashfe-block_1|cashfe-block_2|autogluon), (\w+): (-?\d+\.\d+), (-?\d+\.\d+)", line)
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
file_path = './results.txt'
results, ranks = parse_data(file_path)

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + rank_fields
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
