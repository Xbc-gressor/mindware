import json
import numpy as np
from scipy.stats import kendalltau, spearmanr

def cal_cor(file_path, topk=1):
    with open(file_path, 'r') as f:
        file = json.load(f)
    board = [tmp.split(', ') for tmp in file["leader_board"]]
    train_score = np.array([float(tmp[0].split('train-')[1]) for tmp in board])
    train2_score = np.array([float(tmp[1].split('train_2-')[1]) for tmp in board])
    val_score = np.array([float(tmp[4].split('val-')[1]) for tmp in board])
    val2_score = np.array([float(tmp[5].split('val_2-')[1]) for tmp in board])
    test_score = np.array([float(tmp[2].split('test-')[1]) for tmp in board])

    topn = int(len(test_score) * topk)
    sel_idx = np.argsort(-test_score)[:topn]

    cols = []
    cols.append(spearmanr(train_score[sel_idx], test_score[sel_idx])[0])
    cols.append(spearmanr(train2_score[sel_idx], test_score[sel_idx])[0])
    cols.append(spearmanr(val_score[sel_idx], test_score[sel_idx])[0])
    cols.append(spearmanr(val2_score[sel_idx], test_score[sel_idx])[0])
    cols.append(len(board))

    return cols

def sel(file_path, topn=10):
    with open(file_path, 'r') as f:
        file = json.load(f)
    board = [tmp.split(', ') for tmp in file["leader_board"]]
    train_score = np.array([float(tmp[0].split('train-')[1]) for tmp in board])
    train2_score = np.array([float(tmp[1].split('train_2-')[1]) for tmp in board])
    val_score = np.array([float(tmp[4].split('val-')[1]) for tmp in board])
    val2_score = np.array([float(tmp[5].split('val_2-')[1]) for tmp in board])
    test_score = np.array([float(tmp[2].split('test-')[1]) for tmp in board])

    scores = []
    for tmp in [train_score, train2_score, val_score, val2_score]:
        sel_idx = np.argsort(-tmp)[:topn]
        scores.append(np.mean(test_score[sel_idx]))
    ranks = list(np.argsort(-np.array(scores)) + 1)
    ranks.append(len(board))
    return ranks


# def compare(file_path, topk=0.3):
#     with open(file_path, 'r') as f:
#         file = json.load(f)
#     board = [tmp.split(', ') for tmp in file["leader_board"]]
#     test_score = np.array([float(tmp[2].split('test-')[1]) for tmp in board])

#     topn = int(len(test_score) * topk)
#     sel_idx = np.argsort(-test_score)[:topn]

#     test_score = test_score[sel_idx]
#     head = [board[i][0].split(': ')[0] for i in sel_idx]

#     layer = [int(tmp.split('-')[2][1:]) for tmp in head]
#     ranks = list(np.argsort(-np.array(test_score)) + 1)
#     layer_dict = {}
#     for i in range(len(layer)):
#         l = layer[i]
#         r = ranks[i]
#         if l not in layer_dict:
#             layer_dict[l] = []
#         layer_dict[l].append(r)

#     for key in layer_dict.keys():
#         layer_dict[key] = np.mean(layer_dict[key])

#     return [layer_dict.get(tmp, 100000) for tmp in [1,2,3,4,5]]


def compare(file_path, topk=1):
    with open(file_path, 'r') as f:
        file = json.load(f)
    board = [tmp.split(', ') for tmp in file["leader_board"]]
    test_score = np.array([float(tmp[2].split('test-')[1]) for tmp in board])

    topn = int(len(test_score) * topk)
    sel_idx = np.argsort(-test_score)[:topn]

    test_score = test_score[sel_idx]
    head = [board[i][0].split(': ')[0] for i in sel_idx]

    layer = [int(tmp.split('-')[0].split('_')[0][3:]) for tmp in head]
    # layer = [int(tmp.split('-')[0].split('_')[1][1:]) for tmp in head]
    ranks = list(np.argsort(-np.array(test_score)) + 1)
    layer_dict = {}
    for i in range(len(layer)):
        l = layer[i]
        r = ranks[i]
        if l not in layer_dict:
            layer_dict[l] = []
        layer_dict[l].append(r)

    for key in layer_dict.keys():
        layer_dict[key] = np.mean(layer_dict[key])

    return [layer_dict.get(tmp, np.nan) for tmp in range(4, 50, 4)]

path_lists = [
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_ailerons_2025-04-28-01-36-08-029173/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-04-28-03-06-56-426518/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_kc1_2025-04-28-01-36-08-409515/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_mv_2025-04-28-04-06-39-026878/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_sick_2025-04-28-02-45-22-638443/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_cpu_act_2025-04-28-01-36-07-994132/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_debutanizer_2025-04-28-03-05-08-744295/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_Moneyball_2025-04-28-01-36-08-142554/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_puma8NH_2025-04-28-03-57-29-681361/best_model_info.json",
    "/root/mindware/examples/benchmark/res_ensopt_dropout_data_maxk/ENS-smac(1)-cv_bank32nh_2025-04-28-04-22-12-111699/best_model_info.json",

]


# cor_list = []
# for path in path_lists:
#     cor_list.append(cal_cor(path, 1))

# for i in range(len(cor_list)):
#     tmp = cor_list[i]
#     print("%3d" % i, end=': ')
#     for t in tmp[:-1]:
#         print("%7.4f" % t, end=', ')
#     print("%5d" % tmp[-1])
# cor_mean = np.mean(cor_list, axis=0)
# print("avg", end=': ')
# for t in cor_mean[:-1]:
#     print("%7.4f" % t, end=', ')
# print("%5d" % cor_mean[-1])

"""

top 50%
  0:  0.0435,  0.1302, -0.0632,  0.0225,   166
  1: -0.2299, -0.1988,  0.0118,  0.0648,   449
  2: -0.1458, -0.3149,  0.3748,  0.0597,   974
  3:  0.0271, -0.0313,  0.3935,  0.1946,    93
  4: -0.1785,  0.2122,  0.3066,  0.1469,   349
  5:  0.2000,  0.2000,  0.2307,  0.2307,   173
  6: -0.1569, -0.1569, -0.3853, -0.3853,    80
  7:  0.3130,  0.3130,  0.1925,  0.1925,  2349
  8:  0.1507,  0.1507,  0.1463,  0.1463,   416
  9:  0.7862,  0.7862, -0.1373, -0.1373,    84
avg:  0.0809,  0.1090,  0.1070,  0.0535,    84

top 30%
  0: -0.1556, -0.1022, -0.1690, -0.3521,   166
  1: -0.1928, -0.1414,  0.1007,  0.0447,   449
  2: -0.1187, -0.3053,  0.4360,  0.1366,   974
  3: -0.1476,  0.0909,  0.1452,  0.1638,    93
  4: -0.5558,  0.2451,  0.6761,  0.2800,   349
  5:  0.0644,  0.0644,  0.1795,  0.1795,   173
  6: -0.2294, -0.2294,  0.0201,  0.0201,    80
  7:  0.2228,  0.2228,  0.1425,  0.1425,  2349
  8: -0.0574, -0.0574,  0.2030,  0.2030,   416
  9:  0.7266,  0.7266,  0.4307,  0.4307,    84
avg: -0.0443,  0.0514,  0.2165,  0.1249,    84

top 10%
  0: -0.3899, -0.3155,  0.2938, -0.2549,   166
  1: -0.0685, -0.0481,  0.1710,  0.1117,   449
  2: -0.0892, -0.0965,  0.2268,  0.0672,   974
  3: -0.3819,  0.0000,  0.4009,  0.1081,    93
  4: -0.0990,  0.2336,  0.1178,  0.3004,   349
  5: -0.5558, -0.5558, -0.3021, -0.3021,   173
  6: -0.4849, -0.4849, -0.1576, -0.1576,    80
  7:  0.0035,  0.0035,  0.0902,  0.0902,  2349
  8:  0.3711,  0.3711, -0.0341, -0.0341,   416
  9:  0.0870,  0.0870, -0.0370, -0.0370,    84
avg: -0.1607, -0.0805,  0.0770, -0.0108,    84
"""


# 根据score选取topn个，看看test score的平均值是多少
# cor_list = []
# for path in path_lists:
#     cor_list.append(sel(path, 10))

# for i in range(len(cor_list)):
#     tmp = cor_list[i]
#     print("%3d" % i, end=': ')
#     for t in tmp[:-1]:
#         print("%5d" % t, end=', ')
#     print("%5d" % tmp[-1])
# cor_mean = np.mean(cor_list, axis=0)
# print("avg", end=': ')
# for t in cor_mean[:-1]:
#     print("%5.2f" % t, end=', ')
# print("%5d" % cor_mean[-1])
"""
top10
cor_list = []
for path in path_lists:
    cor_list.append(sel(path, 10))

for i in range(len(cor_list)):
    tmp = cor_list[i]
    print("%3d" % i, end=': ')
    for t in tmp[:-1]:
        print("%5d" % t, end=', ')
    print("%5d" % tmp[-1])
cor_mean = np.mean(cor_list, axis=0)
print("avg", end=': ')
for t in cor_mean[:-1]:
    print("%5.2f" % t, end=', ')
print("%5d" % cor_mean[-1])
"""

# 看不同head的test_score
cor_list = []
for path in path_lists:
    cor_list.append(compare(path, topk=0.8))

for i in range(len(cor_list)):
    tmp = cor_list[i]
    print("%3d" % i, end=': ')
    for t in tmp:
        if not np.isnan(t):
            print("%8d" % t, end=', ')
        else:
            print("%8s" % "nan", end=', ')
    print()
cor_mean = np.nanmean(cor_list, axis=0)
print("avg", end=': ')
for t in cor_mean:
    if not np.isnan(t):
        print("%8.2f" % t, end=', ')
    else:
        print("%8s" % "nan", end=', ')
print()