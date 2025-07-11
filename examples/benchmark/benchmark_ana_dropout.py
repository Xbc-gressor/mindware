import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau
import math

data_dir = './benchmark_data'

data_dict = {
    "CLS":{}, "RGS":{},
}

sel_ens = [0, 0.1, 0.2, 0.3, 0.4]

for task_type in ["CLS", "RGS"]:
    tt = 3600 if task_type == 'CLS' else 3600
    sub_dir = os.path.join('./benchmark_data/', f'results_{task_type}_{tt}')
    for _result_file in os.listdir(sub_dir):
        result_file = os.path.join(sub_dir, _result_file)

        print(_result_file)
        with open(result_file, 'r') as f:
            results = json.load(f)

        if 'dropoutreg_exp' not in results or results['dropoutreg_exp'] == {}:
            continue

        task_id = _result_file[:-5]
        if task_type == 'RGS':
            if task_id in ['plasma_retinol', 'mbagrade', 'pollen', 'space_ga', 'elusage', 'bank8FM', 'Crash', 'pyrim', 'disclosure_x_noise', 'strikes', 'weather_izmir', 'fri_c3_100_25', 'cloud', 'carprice']:
                continue

        for size_ratio in results['dropoutreg_exp']:
            dropout = float(size_ratio.split('dropout')[1])
            if dropout not in sel_ens:
                continue

            if task_id not in data_dict[task_type]:
                data_dict[task_type][task_id] = {}

            # 记录base model的质量
            if task_type == 'CLS':
                coef = np.array(results['dropoutreg_exp'][size_ratio][0])
                m, n = coef.shape
                try:
                    coef = coef.reshape(m, 30, n//30)
                    coef = np.sum(coef, axis=2)
                except:
                    pass
            else:
                coef = np.array([results['dropoutreg_exp'][size_ratio][0]])

            # data_dict[task_type][task_id][dropout] = np.std(coef)
            res = []
            for tmp in coef:
                # tmp = np.abs(tmp)
                idx = np.argsort(tmp)
                res.append(tmp[idx[:1]].sum() / np.sum(np.abs(tmp)))
                # res.append(np.std(tmp))
            
            data_dict[task_type][task_id][dropout] = np.mean(res)



from prettytable import PrettyTable

tar_key = 'ALL'
imp_dict = []
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in data_dict.items():

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
            imp_dict.append(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)



if not os.path.exists('./images/dropout/'):
    os.mkdir('./images/dropout/')

tar_key = 'RGS'
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(range(len(imp_dict)), imp_dict, marker='o', linewidth=2.5)
# Customize plot
plt.xlabel('Dropout Ratio', fontsize=22)
plt.ylabel('Maximum Weight Proportion', fontsize=22)
# plt.ylabel('Weight Std', fontsize=22)
# plt.title('Normalized Score of Base Models')
plt.legend(fontsize=18)
plt.xticks(range(len(sel_ens)), sel_ens, fontsize=20)
plt.tight_layout()

# Show plot
plt.savefig('./images/dropout/std_with_dropout.png')
plt.show()
