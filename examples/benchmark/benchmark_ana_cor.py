import os
import json
import shutil
import numpy as np
from scipy.stats import spearmanr, kendalltau

data_dir = './benchmark_data'
sel_ens = [0, 20]

data_dict = {"CLS":{}, "RGS":{}}
opt_data_dict = {"CLS":{}, "RGS":{}}

for sub_dir in ['data_CLS_3600', 'data_RGS_7200']:
    sub_dir = os.path.join(data_dir, sub_dir)
    for sub2_dir in os.listdir(sub_dir):
        sub2_dir = os.path.join(sub_dir, sub2_dir)

        for sub3_dir in sorted(os.listdir(sub2_dir)):
            if not sub3_dir.startswith('ENS'):
                continue
            sub3_dir = os.path.join(sub2_dir, sub3_dir)
            config_path = os.path.join(sub3_dir, './config.json')
            best_path = os.path.join(sub3_dir, './best_model_info.json')
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

            if config['time_limit'] != 1 and len(tar_dict[task_type][task_id][d]) > 1:
                continue
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
                sub_len = len(tmp_list) // 2
                train_score = [float(tmp[0].split('train-')[1]) for tmp in tmp_list][:sub_len]
                val_score = [float(tmp[4].split('val-')[1]) for tmp in tmp_list][:sub_len]
                test_score = [float(tmp[2].split('test-')[1]) for tmp in tmp_list][:sub_len]
                cor = spearmanr(train_score, test_score)[0]
                cor = spearmanr(val_score, test_score)[0]
                cor_dict[task_type][_dataset][dropout] = cor
        except:
            if _dataset in cor_dict[task_type]:
                cor_dict[task_type].pop(_dataset)
            print(f"数据集{_dataset}数据不全！")


fields = dict(
    size_ratios = ['ens20_r40_'],  # , 'ens10_r20_', 'ens20_r40_', 'ens20_r20_'
    dropouts = [0, 20],
    layers = ['L1', 'L2', 'L3', 'L4'],
    heads = ['weighted', 'linear'],
)
tar_str = 'dropouts'

# if size_ratio in ['ens10_r20_', 'ens20_r40_'] and L == 'L3':
#     continue
ori_strs = list(set(fields.keys()) - {'size_ratios', tar_str})
print(ori_strs)

for task_type, datasets in data_dict.items():
    if task_type not in rank_dict:
        rank_dict[task_type] = {}
    for _dataset, dropouts in datasets.items():
        for size_ratio in fields['size_ratios']:
            for x1 in fields[ori_strs[0]]:
                for x2 in fields[ori_strs[1]]:

                    dataset = f"{_dataset}_{size_ratio}_{x1}_{x2}"
                    if dataset not in rank_dict[task_type]:
                        rank_dict[task_type][dataset] = {}
                    try:
                        val_scores = {}
                        for tar in fields[tar_str]:

                            tmp0 = [x1, x2, tar]
                            tmp1 = [ori_strs[0], ori_strs[1], tar_str]

                            head = tmp0[tmp1.index('heads')]
                            layer = tmp0[tmp1.index('layers')]
                            dropout = tmp0[tmp1.index('dropouts')]

                            leaderboard = dropouts[dropout]
                            ens_name = [tmp.split(': ')[0].split('-') for tmp in leaderboard]
                            tar_idx = None
                            for i in range(len(ens_name)):
                                if ens_name[i][0].startswith(size_ratio) and ens_name[i][1] == head and ens_name[i][2] == layer:
                                    tar_idx = i
                                    break
                            tmp = leaderboard[tar_idx]
                            test_score = float(tmp.split(', ')[2].split('test-')[1])
                            val_score = float(tmp.split(', ')[4].split('val-')[1])
                            rank_dict[task_type][dataset][tar] = test_score
                            val_scores[tar] = val_score

                        max_tar = max(val_scores.items(), key= lambda x: x[1])[0]
                        rank_dict[task_type][dataset]['opt'] = rank_dict[task_type][dataset][max_tar]

                        tmp_dict = rank_dict[task_type][dataset]
                        # 根据值对键进行排序
                        sorted_items = sorted(tmp_dict.items(), key=lambda x: -x[1])

                        # 创建一个字典来存储排名
                        rank = 1
                        # 遍历排序后的项目，为每个键分配排名
                        for i, (key, value) in enumerate(sorted_items):
                            if i > 0 and value != sorted_items[i - 1][1]:
                                rank = i + 1
                            tmp_dict[key] = rank

                        # if task_type == 'CLS':
                        #     tmp_dict[20] = (tmp_dict[20] - tmp_dict[0]) / abs(1-tmp_dict[0]) * 100
                        # else:
                        #     tmp_dict[20] = (tmp_dict[20] - tmp_dict[0]) / abs(tmp_dict[0]) * 100
                        # tmp_dict[0] = 0
                        # if tmp_dict[0] == tmp_dict[20]:
                        #     tmp_dict[0] = 1
                        #     if task_type == 'RGS':
                        #         tmp_dict[0] = 2
                        #     tmp_dict[20] = 1
                        # elif tmp_dict[0] > tmp_dict[20]:
                        #     tmp_dict[0] = 1
                        #     tmp_dict[20] = 2
                        # else:
                        #     tmp_dict[0] = 2
                        #     tmp_dict[20] = 1
                    except Exception as e:
                        if dataset in rank_dict[task_type]:
                            rank_dict[task_type].pop(dataset)
                        print(f"{e}, 数据集{dataset}数据不全！")
sel_ens = fields[tar_str] + ['opt']
# sel_ens = [0, 20]

from prettytable import PrettyTable
table = PrettyTable()
headers = ["Task Type", "Dataset"] + sel_ens

avgs = {
    "CLS": {t:[] for t in sel_ens},
    "RGS": {t:[] for t in sel_ens},
    "ALL": {t:[] for t in sel_ens}
}
table.field_names = headers

for task_type, datasets in rank_dict.items():

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
    for algorithm in algorithms:
        num_dict[task_type] = len(algorithms[algorithm])
        algorithms[algorithm] = np.nanmean(algorithms[algorithm])

    avg_dict[task_type] = algorithms

    table.add_row([task_type, "average"] + ["%.5f" % algorithms[t] for t in sel_ens])

table.add_row(["-"*9, "-"*12] + ["-"*11] * len(sel_ens))
table.add_row(headers)

print(table)
print(num_dict)
breakpoint()