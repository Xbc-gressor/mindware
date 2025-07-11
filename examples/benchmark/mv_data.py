import pickle as pkl
import shutil
import os
import json

meta_res = './benchmark_data'

for file in os.listdir(meta_res):
    if not file.startswith('STA-'):
        continue

    # task_id = file[file.index('holdout')+8:file.index('2025')-1]
    # print(task_id)
    sub_dir = os.path.join(meta_res, file)

    if not os.path.exists(os.path.join(sub_dir, 'config.json')):
        continue

    config = json.load(open(os.path.join(sub_dir, 'config.json'), 'r'))

    task_type = 'data_CLS_3600'
    if config['task_type'] == 4:
        task_type = 'data_RGS_3600'
    task_id = config['task_id']

    target_dir = os.path.join(meta_res, task_type, task_id)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.move(sub_dir, target_dir)

# import numpy as np

# for m in os.listdir(meta_res):
#     sub = os.path.join(meta_res, m)
#     for a in os.listdir(sub):
#         subsub = os.path.join(sub, a)
#         for t in os.listdir(subsub):
#             sub3 = os.path.join(subsub, t)

#             for final in os.listdir(sub3):
#                 sub4 = os.path.join(sub3, final)
#                 if not os.path.exists(os.path.join(sub4, 'best_model_info.json')):
#                     print(sub4)
#                     shutil.rmtree(sub4)

#             ts = np.array(os.listdir(sub3))
#             cat_ts = np.array([t[:t.find('_2025-')] for t in ts])
#             set_ts = set(cat_ts)

#             for sts in set_ts:
#                 idxs = np.where(cat_ts == sts)[0]
#                 if len(idxs) == 1:
#                     continue

#                 sel_ts = ts[idxs]
#                 sel_ts = sel_ts[[1,0]]
#                 sel_ts.sort()

#                 # 只留最后一个
#                 for rm_ts in sel_ts[:-1]:
#                     print(os.path.join(sub3, rm_ts), '留下', sel_ts[-1])
#                     shutil.rmtree(os.path.join(sub3, rm_ts))
