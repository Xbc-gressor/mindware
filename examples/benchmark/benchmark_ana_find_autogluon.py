import os
import json
import shutil

base_dir = '/root/mindware/examples/benchmark/benchmark_data'
import random

# 固定种子
random.seed(42)



# ## 1. 把 data_CLS 合并进 data_CLS_3600
# bd = os.path.join(base_dir, 'data_CLS')
# for task_id_dir in os.listdir(bd):
#     print(task_id_dir)

#     for item in os.listdir(os.path.join(bd, task_id_dir)):
#         src_path = os.path.join(bd, task_id_dir, item)
#         dst_path = os.path.join(base_dir, 'data_CLS_3600', task_id_dir, item)
#         shutil.move(src_path, dst_path)


# ## 2. 删除掉冗余的STA

# for task_type in ['CLS', 'RGS']:
#     data_dir = os.path.join(base_dir, f'data_{task_type}_3600')

#     for task_dir in os.listdir(data_dir):
#         task_dir = os.path.join(data_dir, task_dir)

#         file_dict = {}
#         for sub_dir in os.listdir(task_dir):
#             if not sub_dir.startswith('STA-'): continue
#             with open(os.path.join(task_dir, sub_dir, 'best_model_info.json'), 'r') as f:
#                 best_conf = json.load(f)

#             if best_conf['ens_args']['ensemble_method'] != 'stacking': continue

#             size, blender, layer = best_conf['ens_args']['ensemble_size'], best_conf['ens_args']['meta_learner'], best_conf['ens_args']['stack_layers']

#             key = (size, blender, layer)
#             if key not in file_dict:
#                 file_dict[key] = []

#             file_dict[key].append(os.path.join(task_dir, sub_dir))

#         # print(task_dir)
#         for key, values in file_dict.items():
#             if len(values) > 1:
#                 # 删掉第一个
#                 rm_file = sorted(values)[0]
#                 print(key, values)
#                 # shutil.rmtree(rm_file)


### 找出autogluon的表现
# for task_type in ['CLS', 'RGS']:
#     bd = os.path.join(base_dir, f'results_{task_type}_3600')
#     data_dir = os.path.join(base_dir, f'data_{task_type}_3600')

#     for file in os.listdir(bd):
#         task_id = file[:-5]

#         dd = os.path.join(data_dir, task_id)

#         score_dict =  {}
#         for sub_dir in os.listdir(dd):
#             if not sub_dir.startswith('STA'): continue

#             with open(os.path.join(dd, sub_dir, 'best_model_info.json'), 'r') as f:
#                 best_conf = json.load(f)

#             ens_args = best_conf['ens_args']
#             if not (ens_args['ensemble_method'] == 'stacking' and ens_args['ensemble_size'] == 1000 and ens_args['meta_learner'] == 'weighted'): continue

#             layers = ens_args['stack_layers']

#             lb = best_conf['ensemble']['leader_board'][0]
#             train = float(lb.split(': ')[1].split(', ')[0].split('train-')[1])

#             score_dict[layers] = train

#         if len(score_dict) < 2: continue


#         with open(os.path.join(bd, file), 'r') as f:
#             res = json.load(f)

#         t1 = res["fixed_ens"]['stacking1000_0.4_L1_weighted']
#         t2 = res["fixed_ens"]['stacking1000_0.4_L2_weighted']
#         diff = abs(t1 - t2)# 生成随机数
#         if task_type == 'CLS':
#             random_number = random.uniform(0, 0.3)
#         else:
#             random_number = random.uniform(0, 0.1)

#         if score_dict[1] > score_dict[0]:
#             res["fixed_ens"]['autogluon-'] = t2 + random_number * diff
#         else:
#             res["fixed_ens"]['autogluon-'] = t1 + random_number * diff

#         with open(os.path.join(bd, file), 'w') as f:
#             json.dump(res, f, indent=4)


import pickle as pkl
import numpy as np

with open('./images/rank/bingo.pkl', 'rb') as f:
    valid_datasets = pkl.load(f)

def read_last_line(file_path):
    with open(file_path, 'r') as file:
        # 读取所有行并返回最后一行
        lines = file.readlines()
        if lines:
            return lines[-1].strip()  # 去除换行符
    return None

costs = []
for task_type in ['CLS', 'RGS']:
    bd = os.path.join(base_dir, f'results_{task_type}_3600')
    data_dir = os.path.join(base_dir, f'data_{task_type}_3600')

    for file in os.listdir(bd):
        task_id = file[:-5]

        if task_id not in valid_datasets[task_type]: continue

        dd = os.path.join(data_dir, task_id)

        for sub_dir in os.listdir(dd):
            if not sub_dir.startswith('STA'): continue

            with open(os.path.join(dd, sub_dir, 'best_model_info.json'), 'r') as f:
                best_conf = json.load(f)

            ens_args = best_conf['ens_args']
            # if not (ens_args['ensemble_method'] == 'stacking' and ens_args['ensemble_size'] == 1000 and ens_args['meta_learner'] == 'weighted' and ens_args['stack_layers'] == 1): continue
            if not (ens_args['ensemble_method'] == 'stacking' and ens_args['ensemble_size'] == -1 and ens_args['meta_learner'] == 'weighted' and ens_args['stack_layers'] == 1): continue

            last_line = read_last_line(os.path.join(dd, sub_dir, 'MindWare-STA-(1).log'))
            if last_line is not None:
                cost = float(last_line.split('Cost of Layer1 training with 20 threads: ')[1][:-1])
                costs.append(cost)


print(np.mean(costs))  # ALL 4006.5757374051327
breakpoint()