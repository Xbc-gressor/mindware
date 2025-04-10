import re
import numpy as np
from rgs_benchmark import chosen_datasets_info as rgs_info
from cls_benchmark import chosen_datasets_info as cls_info


# rank_fields = ['cashfe_best', 'cashfe_ens', 'cashfe-block_1_best', 'cashfe-block_1_ens']  # 比较mab和block_1
rank_fields = ['cashfe-block_1-filter_m-1_p-1_ens', 'autogluon_ens']  # 比较mab和block_1 cashfe-block_1_ens, cashfe-block_1-filter_m-1_p-1_ens
# rank_fields = ['cashfe-block_0_ens', 'cashfe-block_1_ens', 'cashfe-block_2_ens']  # 比较mab和block_1
# rank_fields = ['cashfe-block_0_ens', 'cashfe-block_1_ens', 'cashfe-block_2_ens', 'newcashfe-block_1_ens', 'newcashfe-block_2_ens']  # 比较new fe，结果不如原来的
# rank_fields = ['cashfe-block_1_ens', 'newcashfe-block_1_ens', 'newcashfe_rmica-block_1_ens', 'newcashfe_ica-block_1_ens']  # 比较去掉ica后new fe的效果
# rank_fields = ['cashfe-block_1-filter_m6_p6_ens', 'cashfe-block_1-filter_m6_p-1_ens', 'cashfe-block_1-filter_m-1_p-1_ens', 'cashfe-block_2-filter_m6_p6_ens', 'cashfe-block_2-filter_m6_p-1_ens', 'cashfe-block_2-filter_m-1_p-1_ens']
# rank_fields = ['cashfe-block_1-filter_m6_p6_best', 'cashfe-block_1-filter_m6_p-1_best', 'cashfe-block_1-filter_m-1_p-1_best', 'cashfe-block_1-filter_m6_p6_ens', 'cashfe-block_1-filter_m6_p-1_ens', 'cashfe-block_1-filter_m-1_p-1_ens']
# rank_fields = ['cashfe-block_1-filter_m6_p6_ens', 'cashfe-block_1-filter_m6_p-1_ens', 'cashfe-block_1-filter_m-1_p-1_ens']
# rank_fields = ['cashfe-block_2-filter_m6_p6_ens', 'cashfe-block_2-filter_m6_p-1_ens', 'cashfe-block_2-filter_m-1_p-1_ens']
# rank_fields = ['cashfe-block_1_ens', 'cashfe-block_1-filter_m-1_p-1_ens']
rank_fields = ['cashfe-block_1-cv-none-filter_m6_p6-best', 'cashfe-block_1-full-none-filter_m6_p6-best', 'cashfe-block_1-partial-none-filter_m6_p6-best',
               'cashfe-block_1-cv-ensemble_selection10-filter_m6_p6-ens', 'cashfe-block_1-full-ensemble_selection10-filter_m6_p6-ens', 'cashfe-block_1-partial-ensemble_selection10-filter_m6_p6-ens',
               'cashfe-block_1-cv-blending10_0.4-filter_m6_p6-ens', 'cashfe-block_1-full-blending_10_0.4-filter_m6_p6-ens', 'cashfe-block_1-partial-blending10_0.4-filter_m6_p6-ens',
               'cashfe-block_1-cv-stacking10_0.4-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4-filter_m6_p6-ens', 'cashfe-block_1-partial-stacking10_0.4-filter_m6_p6-ens',

               'cashfe-block_1-full-blending_10_0.4_L2-filter_m6_p6-ens', 'cashfe-block_1-full-blending_10_0.4_skipL2-filter_m6_p6-ens', 'cashfe-block_1-full-blending_10_0.4_skipretL2-filter_m6_p6-ens',
               'cashfe-block_1-full-stacking_10_0.4_L2-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipL2-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipretL2-filter_m6_p6-ens',
               'cashfe-block_1-full-stacking_10_0.4_skipretL2we-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipretL2lgb-filter_m6_p6-ens',
               'cashfe-block_1-full-stacking_10_0.4_skipretL3-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipretL4-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipretL5-filter_m6_p6-ens', 
               'cashfe-block_1-full-stacking_10_0.4_skipretL2v5-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_skipretL2v3-filter_m6_p6-ens', 
               'cashfe-block_1-full-stacking_10_0.4_selectL5-filter_m6_p6-ens', 'cashfe-block_1-full-stacking_10_0.4_fullselectL5-filter_m6_p6-ens']
# rank_fields = rank_fields[6:12]
# rank_fields = [rank_fields[7], rank_fields[12], rank_fields[13], rank_fields[14]]  # blending家族：ble、ble L2、ble skip L2、ble skipret L2
# rank_fields = [rank_fields[10],                 rank_fields[16], rank_fields[17]]  # stacking家族：ble、ble L2、ble skip L2、ble skipret L2
# rank_fields = [rank_fields[7], rank_fields[12], rank_fields[13], rank_fields[14], rank_fields[10],                 rank_fields[16], rank_fields[17]]

# rank_fields = [rank_fields[12], rank_fields[13]]  # 比较blending上是否残差

# rank_fields = [rank_fields[7], rank_fields[10]]  # 比较blending和stacking
# rank_fields = [rank_fields[13], rank_fields[16]]  # 比较blending堆叠和stacking堆叠
# rank_fields = [rank_fields[7], rank_fields[13], rank_fields[10], rank_fields[16]]
# rank_fields = [rank_fields[14], rank_fields[17]]  # 比较blendingret和stackingret
# rank_fields = [rank_fields[17], rank_fields[18], rank_fields[19]]  # 比较stacking的输出头
# rank_fields = [rank_fields[17], rank_fields[20], rank_fields[21], rank_fields[22]]  # 比较stacking的堆叠层数
# rank_fields = [rank_fields[17], rank_fields[23], rank_fields[24]]  # 比较stacking的cv数量
# rank_fields = [rank_fields[17], rank_fields[20], rank_fields[21], rank_fields[22], rank_fields[25], rank_fields[26]]
# rank_fields = [rank_fields[25], rank_fields[26]]

# rank_fields = ['autogluon-ens', rank_fields[24]]

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
            match = re.match(r"(CLS|RGS): ([^,]+), (\w+): (-?\d+\.\d+), (-?\d+\.\d+|None)", line)
            if match:
                task_type, algorithm, dataset, best, ens = match.groups()
                if dataset not in results[task_type]:
                    results[task_type][dataset] = {}
                results[task_type][dataset][f"{algorithm}-best"] = float(best)
                if ens != 'None':
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
file_path = './res_refitcv_data.txt'
results, ranks = parse_data(file_path)

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"]

for tmp in rank_fields:
    print(tmp)
    tmp_list = tmp.split('-')
    if 'best' in tmp:
        tmp_list = [tmp_list[2], tmp_list[-1]]
    elif 'autogluon' in tmp:
        tmp_list = ['autogluon']
    else:
        ens = tmp_list[3]
        if 'selection' in ens:
            ens = 'sel'
        else:
            if 'L' in ens:
                ens = ens[:3] + '_' + ens.split('_')[-1]
            else:
                ens = ens[:3]
        tmp_list = [tmp_list[2], ens, tmp_list[-1]]
    headers.append('-'.join(tmp_list))
    
avgs = {
    "CLS": {t:[] for t in rank_fields},
    "RGS": {t:[] for t in rank_fields},
    "ALL": {t:[] for t in rank_fields}
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
    for algorithm in algorithms:
        algorithms[algorithm] = np.mean(algorithms[algorithm])
    
    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in rank_fields])
        

print(table)
