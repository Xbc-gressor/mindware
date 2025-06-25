import os
import json
import shutil

data_dir = './benchmark_data'

path_dict = {0:{}, 4:{}}

for sub_dir in os.listdir(data_dir):
    # if sub_dir.startswith('CASHFE') or sub_dir.startswith('results'):
    #     continue
    if not sub_dir.startswith('CMA_ES-'): continue
    config_path = os.path.join(data_dir, sub_dir, './config.json')
    best_path = os.path.join(data_dir, sub_dir, './best_model_info.json')
    if not os.path.exists(config_path) or not os.path.exists(best_path):
        continue

    with open(config_path, 'r') as f:
        config = json.load(f)

    if config['time_limit'] != 1:
        continue

    with open(best_path, 'r') as f:
        best_config = json.load(f)

    task_type = config['task_type']
    task_id = config['task_id']
    if task_id not in path_dict[task_type]:
        path_dict[task_type][task_id] = {}
    best_one = best_config["best_pool"][0]
    # a, b, c = best_one["ensemble_size"], best_one["ratio"], best_one["dropout"]
    # ens_str = f"{a}_{b}_{c}"
    # ens_str = config['time_limit']
    ens_str = f"{config['ensemble_method']}_{config['size']}"
    if ens_str not in path_dict[task_type][task_id]:
        path_dict[task_type][task_id][ens_str] = []

    path_dict[task_type][task_id][ens_str].append(sub_dir)


for task_type in path_dict:
    for task_id in path_dict[task_type]:
        tmp = path_dict[task_type][task_id]
        for ens in tmp:
            if len(tmp[ens]) > 1:
                print(task_type, task_id, ens, len(tmp[ens]))
                sort_list = sorted(tmp[ens])
                print(sort_list)
                # shutil.rmtree(os.path.join(data_dir, sort_list[0]))
breakpoint()
