import pickle as pkl
import shutil
import os
import json

task = 'rgs'

metrics = ['acc', 'f1', 'auc']
metrics_map = {'acc': 'acc', 'auc': 'auc', 'f1': 'f1', "make_scorer(accuracy_score)": 'acc'}
if task == 'rgs':
    metrics = ['mse', 'r2', 'mae']
    metrics_map = {
        'make_scorer(mean_squared_error, greater_is_better=False)': 'mse',
        'make_scorer(mean_absolute_error, greater_is_better=False)': 'mae',
        'make_scorer(r2_score)': 'r2',
    }


algorithms = [
    'adaboost',
    'extra_trees',
    'gradient_boosting',
    'k_nearest_neighbors',
    'lda',
    'liblinear_svc',
    'libsvm_svc',
    'lightgbm',
    'logistic_regression',
    'qda',
    'random_forest',
    'xgboost'
]

if task == 'rgs':
    algorithms = [
        'adaboost',
        'extra_trees',
        'gradient_boosting',
        'k_nearest_neighbors',
        'lasso_regression',
        'liblinear_svr',
        'libsvm_svr',
        'lightgbm',
        'random_forest',
        'ridge_regression',
        'xgboost'
    ]


meta_res = '/root/mindware/scripts/data_cls/data'
if task == 'rgs':
    meta_res = '/root/mindware/scripts/data_rgs/data'

# dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_embedding.pkl', 'rb'))
# if task == 'rgs':
#     dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_embedding.pkl', 'rb'))

# task_ids = [t[5:] for t in dataset_embedding['task_ids']]

# for file in os.listdir(meta_res):
#     sub_dir = os.path.join(meta_res, file)

#     if not os.path.exists(os.path.join(sub_dir, 'config.json')):
#         continue

#     config = json.load(open(os.path.join(sub_dir, 'config.json'), 'r'))

#     task_id = config['task_id']
#     algo = config['include_algorithms'][0]
#     metric = config['metric']

#     if metric not in metrics_map:
#         continue
#     metric = metrics_map[metric]

#     target_dir = os.path.join(meta_res, metric, algo, task_id)

#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     shutil.move(sub_dir, target_dir)

import numpy as np

for m in os.listdir(meta_res):
    sub = os.path.join(meta_res, m)
    for a in os.listdir(sub):
        subsub = os.path.join(sub, a)
        for t in os.listdir(subsub):
            sub3 = os.path.join(subsub, t)

            for final in os.listdir(sub3):
                sub4 = os.path.join(sub3, final)
                if not os.path.exists(os.path.join(sub4, 'best_model_info.json')):
                    print(sub4)
                    shutil.rmtree(sub4)

            ts = np.array(os.listdir(sub3))
            cat_ts = np.array([t[:t.find('_2025-')] for t in ts])
            set_ts = set(cat_ts)

            for sts in set_ts:
                idxs = np.where(cat_ts == sts)[0]
                if len(idxs) == 1:
                    continue

                sel_ts = ts[idxs]
                sel_ts = sel_ts[[1,0]]
                sel_ts.sort()

                # 只留最后一个
                for rm_ts in sel_ts[:-1]:
                    print(os.path.join(sub3, rm_ts), '留下', sel_ts[-1])
                    shutil.rmtree(os.path.join(sub3, rm_ts))
