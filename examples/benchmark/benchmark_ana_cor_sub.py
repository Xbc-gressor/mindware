import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau

data_dir = './benchmark_data'
sel_ens = [20]

data_path_dict = [
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_ailerons_2025-05-09-17-59-32-356337/',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_bank32nh_2025-05-09-20-57-50-315934/',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-05-09-17-59-32-290140',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-05-09-19-41-37-953602',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_debutanizer_2025-05-09-19-38-01-715415',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_kc1_2025-05-09-17-59-32-751841',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_Moneyball_2025-05-09-17-59-32-444422',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_mv_2025-05-09-20-37-44-855607',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_puma8NH_2025-05-09-20-30-52-474765',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_sick_2025-05-09-19-13-38-843859'
]

data_path_dict = [
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_ailerons_2025-05-10-11-00-32-156864',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_bank32nh_2025-05-10-14-00-24-751453',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-05-10-11-00-32-112640',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-05-10-12-39-11-493134',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_debutanizer_2025-05-10-12-32-47-106061',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_kc1_2025-05-10-11-00-32-511556',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_Moneyball_2025-05-10-11-00-32-258461',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_mv_2025-05-10-13-38-10-931808',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_puma8NH_2025-05-10-13-24-24-047920',
    '/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_sick_2025-05-10-12-07-08-665399'
]

# data_path_dict = [
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_ailerons_2025-06-08-03-48-29-730898',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_bank32nh_2025-06-08-07-09-59-512769',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_cpu_act_2025-06-08-03-48-29-611355',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_cpu_act_2025-06-08-06-07-37-670000',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_debutanizer_2025-06-08-05-08-35-670593',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_kc1_2025-06-08-03-48-29-629734',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_Moneyball_2025-06-08-05-06-58-195576',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_mv_2025-06-08-04-51-46-393847',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_puma8NH_2025-06-08-05-09-50-190198',
#     '/root/mindware/examples/benchmark/benchmark_data/ENS-smac(1)-cv_sick_2025-06-08-03-48-29-781640'
# ]

data_dict = {"CLS":{}, "RGS":{}}
opt_data_dict = {"CLS":{}, "RGS":{}}

for sub_dir in data_path_dict:
    config_path = os.path.join(sub_dir, './config.json')
    best_path = os.path.join(sub_dir, './best_model_info.json')
    if not os.path.exists(config_path) or not os.path.exists(best_path):
        continue

    with open(config_path, 'r') as f:
        config = json.load(f)

    if config['time_limit'] == 1:
        tar_dict = data_dict
    else:
        tar_dict = opt_data_dict

    with open(best_path, 'r') as f:
        best_config = json.load(f)

    task_type = "CLS"
    if config['task_type'] == 4:
        task_type = "RGS"
    task_id = config['task_id']
    if task_id not in tar_dict[task_type]:
        tar_dict[task_type][task_id] = {}
    d = best_config["best_pool"][0]["dropout"]
    if d not in tar_dict[task_type][task_id]:
        tar_dict[task_type][task_id][d] = []
    tar_dict[task_type][task_id][d] += best_config["leader_board"]

cor_dict = {}
rank_dict = {}

for task_type, datasets in opt_data_dict.items():
    if task_type not in cor_dict:
        cor_dict[task_type] = {}
    for _dataset, dropouts in datasets.items():

        if _dataset not in cor_dict[task_type]:
            cor_dict[task_type][_dataset] = {}
        try:
            for dropout in sel_ens:
                leaderboard = dropouts[dropout]
                tmp_list = [tmp.split(', ') for tmp in leaderboard]
                sub_len = len(tmp_list)
                train_score = [float(tmp[0].split('train-')[1]) for tmp in tmp_list][:sub_len]
                val_score = [float(tmp[4].split('val-')[1]) for tmp in tmp_list][:sub_len]
                test_score = [float(tmp[2].split('test-')[1]) for tmp in tmp_list][:sub_len]

                # tmp_score = [train_score[i] + val_score[i] for i in range(sub_len)]
                cor = spearmanr(train_score, test_score)[0]
                # cor = spearmanr(train_score, test_score)[0]
                cor_dict[task_type][_dataset][dropout] = cor
        except Exception as e:
            if _dataset in cor_dict[task_type]:
                cor_dict[task_type].pop(_dataset)
            print(f"{e}, 数据集{_dataset}数据不全！")



from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in cor_dict.items():

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

avg_dict = {}
for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
breakpoint()