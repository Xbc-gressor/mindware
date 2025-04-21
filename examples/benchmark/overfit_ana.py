
import os
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS     # export NUMEXPR_NUM_THREADS=1

import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import numpy as np
from openbox.utils.history import History, Observation
import pickle as pkl
from scipy.stats import kendalltau, spearmanr

data_dir = './res_ensopt_data'

# data_dict = {'CLS': {}, 'RGS': {}}

# for task in sorted(os.listdir(data_dir)):
#     task_path = os.path.join(data_dir, task)

#     with open(os.path.join(task_path, 'config.json'), 'r') as f:
#         config = json.load(f)

#     with open(os.path.join(task_path, 'best_model_info.json'), 'r') as f:
#         best_info = json.load(f)

#     task_type = config['task_type']
#     task_id = config['task_id']
#     tar_dict = data_dict['CLS'] if task_type == 0 else data_dict['RGS']
#     leader_board = best_info['leader_board']

#     X = []
#     y = []

#     for board in leader_board:
#         # "ens10_r40-weighted-L4: train-0.88874, test-0.89091, val-0.88512"
#         tmp = board.split(': ')
#         head = tmp[0]
#         perfs = tmp[1]

#         tmp = head.split('-')
#         size, ratio = tmp[0].split('_')
#         size = int(size[3:])
#         ratio = int(ratio[1:])
#         h = tmp[1].split('_')[0]
#         layer = int(tmp[2][1:])

#         tmp = perfs.split(', ')
#         train = float(tmp[0][6:])
#         train2 = float(tmp[1][8:])
#         test = float(tmp[2][5:])
#         test2 = float(tmp[3][7:])
#         val = float(tmp[4][4:])
#         val2 = float(tmp[5][6:])

#         X.append([size, ratio, h, layer, train, train2, val, val2])
#         y.append(test)

#     tar_dict[task_id] = (X, y)


# with open('./leader_board.pkl', 'wb') as f:
#     pkl.dump(data_dict, f)


def test(data_dict, cal):

    result_dict = {}
    for t, tar_dict in data_dict.items():
        if t not in result_dict:
            result_dict[t] = {}

        for task, data in tar_dict.items():
            X, y = data
            best_x = None
            best_idx = -1
            for idx, x in enumerate(X):
                if best_x is None:
                    best_x = x
                    best_idx = idx
                else:
                    if cal(x, best_x):
                        best_x = x
                        best_idx = idx

            sel_x, sel_y = X[best_idx], y[best_idx]

            rank = len(np.where(np.array(y) > sel_y)[0]) + 1

            result_dict[t][task] = (sel_x, sel_y, rank)

    return result_dict


# size, ratio, h, layer, train, train2, val, val2
def only_val(x, y):
    return x[6] > y[6]

def only_val2(x, y):
    return x[7] > y[7]

def only_train(x, y):
    return x[4] > y[4]

def val_train(x, y):
    if x[6] != y[6]:
        return x[6] > y[6]
    else:
        return x[4] > y[4]

def val_p_train(x, y):

    return x[4] + x[6] > y[4] + y[6]

def val_val2(x, y):
    if x[6] != y[6]:
        return x[6] > y[6]
    else:
        return x[7] > y[7]

with open('./leader_board.pkl', 'rb') as f:
    data_dict = pkl.load(f)

candidate_cals = {
    'only_train': only_train,
    'only_val': only_val,
    'only_val2': only_val2,
    'val_train': val_train,
    'val_p_train': val_p_train,
    'val_val2': val_val2
}
rank_fields = list(candidate_cals.keys())

final_dict = {}
for mth, cal in candidate_cals.items():
    result_dict = test(data_dict, cal)

    for task_type, datasets in result_dict.items():
        if task_type not in final_dict:
            final_dict[task_type] = {}

        for dataset, res in datasets.items():
            if dataset not in final_dict[task_type]:
                final_dict[task_type][dataset] = {}
            final_dict[task_type][dataset][mth] = res[-1]

from prettytable import PrettyTable
table = PrettyTable()
corrs = ['train_cor', 'train2_cor', 'val_cor', 'val2_cor', 'val_p_val2']
headers = ["Task Type", "Dataset"] + rank_fields + corrs
avgs = {
    "CLS": {t:[] for t in headers[2:]},
    "RGS": {t:[] for t in headers[2:]},
    "ALL": {t:[] for t in headers[2:]}
}

for task_type, datasets in final_dict.items():

    table.field_names = headers

    # 填充表格行数据
    for dataset, res in datasets.items():
        row = [task_type, dataset] + [res[t] for t in rank_fields]

        for t in res:
            avgs[task_type][t].append(res[t])
            avgs["ALL"][t].append(res[t])

        # 计算一下相关性
        raw_X, raw_y = data_dict[task_type][dataset]
        for i in range(5):
            if i == 4:
                tmp_X = [tmp[6] + tmp[7] for tmp in raw_X]
            else:
                tmp_X = [tmp[i+4] for tmp in raw_X]
            ken = round(kendalltau(tmp_X, raw_y)[0], 5)
            row.append(ken)
            avgs[task_type][corrs[i]].append(ken)
            avgs["ALL"][corrs[i]].append(ken)
        table.add_row(row)

    table.add_row(["-"*9, "-"*12] + ["-"*11] * len(headers[2:]))

for task_type, algorithms in avgs.items():
    for algorithm in algorithms:
        algorithms[algorithm] = np.mean(algorithms[algorithm])
    
    table.add_row([task_type, "average"] + ["%.3f" % algorithms[t] for t in headers[2:]])

print(table)

# # 测试代理模型
# from mindware.components.config_space.cs_builder import get_ens_cs
# from openbox.core.base import build_surrogate
# from scipy.stats import kendalltau, spearmanr
# from openbox.utils.config_space.space_utils import get_config_from_dict

# from sklearn.model_selection import KFold
# kf =  KFold(n_splits=10, shuffle=True, random_state=1)

# space = get_ens_cs()

# result_dict = {}
# for t, tar_dict in data_dict.items():
#     if t not in result_dict:
#         result_dict[t] = {}

#     for task, data in tar_dict.items():
#         _X, _y = data
#         print(task, len(_y))
#         his = History(task_id='test', config_space=space)
#         for i in range(len(_X)):
#             x = _X[i]
#             config = get_config_from_dict(space, {'ensemble_size': x[0], 'ratio': x[1], 'meta_learner': x[2], 'stack_layers': x[3]-1})
#             obs = Observation(config=config, objectives=[-_y[i]])
#             his.update_observation(obs)
        
#         X = his.get_config_array()
#         y = his.get_objectives()
#         preds = np.zeros(len(y))
#         for train_idx, val_idx in kf.split(X, y):
#             train_X, train_y = X[train_idx], y[train_idx]
#             val_X, val_y = X[val_idx], y[val_idx]

#             surrogate = build_surrogate(func_str='gp', config_space=space, rng=np.random.RandomState(1))
#             surrogate.train(train_X, train_y)
#             pred, _ = surrogate.predict(val_X)
#             preds[val_idx] = pred.reshape(-1)

#         ken, _ = kendalltau(preds, y)
#         spear, _ = spearmanr(preds, y)
#         result_dict[t][task] = (ken, spear)
#         print(result_dict[t][task])


"""
{'CLS': {'mv': 0.36223742141500603, 'sick': 0.7156595859638313, 'kc1': 0.6018443450672613, 'ailerons': 0.4596847891312104, 'cpu_act': 0.46458536513087934}, 
'RGS': {'cpu_act': 0.6313358555671471, 'bank32nh': 0.7607032716874081, 'Moneyball': 0.7912842640663619, 'debutanizer': 0.6538525829681837, 'puma8NH': 0.6004776669739377}}

GP
mv 138
(0.4136658612171899, 0.5233585950049573)
sick 530
(0.7521261884846323, 0.8714726554610034)
kc1 242
(0.5935503025362225, 0.7638387324046693)
ailerons 481
(0.4542484312028197, 0.6079279664766588)
cpu_act 552
(0.4676557063070832, 0.6188413917295609)
cpu_act 159
(0.5286834704690552, 0.7131151028090693)
bank32nh 216
(0.8301781571896802, 0.951991149737058)
Moneyball 4768
(0.6621903936463813, 0.8456242474960081)
debutanizer 283
(0.66388201227308, 0.825599412618655)
puma8NH 219
(0.6528269512316159, 0.8444731975187664)
"""
