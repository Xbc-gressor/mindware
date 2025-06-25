import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau

data_dir = './benchmark_data'

data_dict = {
    'True': {"CLS":{}, "RGS":{}},
    'False': {"CLS":{}, "RGS":{}}
}


opt_data_dict = {
    0: {"CLS":{}, "RGS":{}},
    1: {"CLS":{}, "RGS":{}},
    2: {"CLS":{}, "RGS":{}},
    3: {"CLS":{}, "RGS":{}},
    4: {"CLS":{}, "RGS":{}}
}


for task_type in ["CLS", "RGS"]:
    tt = 3600 if task_type == 'CLS' else 3600
    sub_dir = os.path.join('./benchmark_data/', f'results_{task_type}_{tt}')
    for _result_file in os.listdir(sub_dir):
        result_file = os.path.join(sub_dir, _result_file)

        print(_result_file)
        with open(result_file, 'r') as f:
            results = json.load(f)

        if 'retain_exp' not in results or results['retain_exp'] == {}:
            continue

        task_id = _result_file[:-5]

        for size_ratio in results['retain_exp']:
            tmp = size_ratio.split('_')
            size, ratio, retain = int(tmp[0][8:]), float(tmp[1]), tmp[4][6:]

            if task_id not in data_dict[retain][task_type]:
                data_dict[retain][task_type][task_id] = {}

            # 记录base model的质量
            layer_loss = results['retain_exp'][size_ratio][0]

            base_loss = np.array(layer_loss[0]['test'])

            data_dict[retain][task_type][task_id][0] = 0

            for layer in range(1, 5):
                if layer > len(layer_loss) - 1:
                    data_dict[retain][task_type][task_id][layer] = data_dict[retain][task_type][task_id][layer-1]
                else:
                    tmp_loss = np.array(layer_loss[layer]['test'])
                    if task_type == 'RGS':
                        tmp_loss = (tmp_loss - base_loss) / np.abs(base_loss) * 100
                    else:
                        tmp_loss = (tmp_loss - base_loss) / np.abs(1 - base_loss + 1e-10) * 100

                    if np.mean(tmp_loss) < -300:
                        data_dict[retain][task_type].pop(task_id)
                        break
                    else:
                        data_dict[retain][task_type][task_id][layer] = np.mean(tmp_loss)

            # 比较rank
            leader_board =  results['retain_exp'][size_ratio][1]

            tmp_data = {}
            for tmp in leader_board:
                meta, scores = tmp.split(': ')
                m, l = meta.split('-')
                l = int(l[1:]) - 1

                test_score = float(scores.split(', ')[2].split('test-')[1])
                if m == 'linear':
                    tmp_data[l] = test_score

            for layer in range(5):

                if task_id not in opt_data_dict[layer][task_type]:
                    opt_data_dict[layer][task_type][task_id] = {}

                if layer not in tmp_data:
                    opt_data_dict[layer][task_type][task_id][retain] = opt_data_dict[layer-1][task_type][task_id][retain]
                else:
                    opt_data_dict[layer][task_type][task_id][retain] = tmp_data[layer]

        for layer in range(5):
            tmp_dict = opt_data_dict[layer][task_type][task_id]
            if tmp_dict['True'] <  tmp_dict['False']:
                tmp_dict['True'] = 2
                tmp_dict['False'] = 1
            elif tmp_dict['True'] >  tmp_dict['False']:
                tmp_dict['True'] = 1
                tmp_dict['False'] = 2
            else:
                tmp_dict['True'] = 1
                tmp_dict['False'] = 1
                if layer == 4:
                    tmp_dict['False'] = 2


sel_ens = [0, 1, 2, 3, 4]

from prettytable import PrettyTable

tar_key = 'ALL'
imp_dict = {
    'True': [],
    'False': []
}
for retain in ['False', 'True']:
    print(retain)
    table = PrettyTable()
    headers = ["Task Type", "Dataset"] + sel_ens

    avgs = {
        "CLS": {t:[] for t in sel_ens},
        "RGS": {t:[] for t in sel_ens},
        "ALL": {t:[] for t in sel_ens}
    }
    table.field_names = headers

    for task_type, datasets in data_dict[retain].items():

        # 填充表格行数据
        for dataset in datasets:
            algorithms = datasets[dataset]
            if algorithms == {}:
                continue
            row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
            table.add_row(row)
            for t in algorithms:
                avgs[task_type][t].append(algorithms[t])
                avgs["ALL"][t].append(algorithms[t])
        table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

    num_dict = {}
    avg_dict = {}
    for task_type, algorithms in avgs.items():
        for algorithm in sorted(algorithms.keys()):
            num_dict[task_type] = len(algorithms[algorithm])
            algorithms[algorithm] = np.nanmean(algorithms[algorithm])

            if task_type == tar_key:
                imp_dict[retain].append(algorithms[algorithm])

        avg_dict[task_type] = algorithms

        table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
    table.add_row(headers)

    print(table)
    print(num_dict)


tar_key = 'ALL'
import matplotlib.pyplot as plt
# Plot
imp_dict['False'][3], imp_dict['False'][4] = imp_dict['False'][4], imp_dict['False'][3]
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(imp_dict['False'])+1), imp_dict['False'], label='no retain', marker='o')
plt.plot(range(1, len(imp_dict['True'])+1), imp_dict['True'], label='retain', marker='s')
plt.axhline(y=0, color='red', linestyle='--')
# Customize plot
plt.xlabel('Stack Layers')
plt.ylabel('Normalized Score of Base Models')
plt.title('Normalized Score of Base Models')
plt.legend()
plt.xticks(range(1, 6))
plt.tight_layout()

# Show plot
plt.savefig('./images/nor_score_with_layers.png')
plt.show()


sel_ens = ['False', 'True']
imp_dict = {
    'True': [],
    'False': []
}
for layer in range(5):
    layer
    table = PrettyTable()
    headers = ["Task Type", "Dataset"] + sel_ens

    avgs = {
        "CLS": {t:[] for t in sel_ens},
        "RGS": {t:[] for t in sel_ens},
        "ALL": {t:[] for t in sel_ens}
    }
    table.field_names = headers

    for task_type, datasets in opt_data_dict[layer].items():

        # 填充表格行数据
        for dataset in datasets:
            algorithms = datasets[dataset]
            if algorithms == {}:
                continue
            row = [task_type, dataset] + ['%.5f' % algorithms[t] for t in sel_ens]
            # table.add_row(row)
            for t in algorithms:
                avgs[task_type][t].append(algorithms[t])
                avgs["ALL"][t].append(algorithms[t])
        table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))

    num_dict = {}
    avg_dict = {}
    for task_type, algorithms in avgs.items():
        for algorithm in sorted(algorithms.keys()):
            num_dict[task_type] = len(algorithms[algorithm])
            algorithms[algorithm] = np.nanmean(algorithms[algorithm])

            if task_type == tar_key:
                imp_dict[algorithm].append(algorithms[algorithm])

        avg_dict[task_type] = algorithms

        table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
    table.add_row(headers)

    print(table)
    print(num_dict)


import matplotlib.pyplot as plt
# Plot
imp_dict['True'][3], imp_dict['True'][4] = imp_dict['True'][4], imp_dict['True'][3]
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(imp_dict['False'])+1), imp_dict['False'], label='no retain', marker='o')
plt.plot(range(1, len(imp_dict['True'])+1), imp_dict['True'], label='retain', marker='s')
# Customize plot
plt.xlabel('Stack Layers')
plt.ylabel('Average Rank')
plt.title('Average Rank of Retain or Not')
plt.legend()
plt.xticks(range(1, 6))
plt.tight_layout()

# Show plot
plt.savefig('./images/rank_with_layers.png')
plt.show()
