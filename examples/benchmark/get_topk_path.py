"""
    获取topk的路径
"""

import os
import json
import glob
import pickle as pkl
from collections import OrderedDict

dataset_embedding_cls = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_embedding.pkl', 'rb'))
cls_rank = [tmp[5:] for tmp in dataset_embedding_cls['task_ids']]
dataset_embedding_rgs = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_embedding.pkl', 'rb'))
rgs_rank = [tmp[5:] for tmp in dataset_embedding_rgs['task_ids']]
# cls_rank = ['kc1', 'sick', 'cpu_act', 'ailerons', 'mv', 'covertype']
# rgs_rank = ['Moneyball', 'debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'black_friday']

def get_task_info(folder):
    config_path = os.path.join(folder, 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            task_id = config.get('task_id')
            task_type = config.get('task_type')
            has_filter_params = 'filter_params' in config and len(config['filter_params']) > 0
            return task_id, task_type, has_filter_params
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        return None, None, None

def get_topk_config_paths(folder):
    return glob.glob(os.path.join(folder, '*topk_config.pkl'))

def main(parent_folder):
    task_data = []

    for subfolder in os.listdir(parent_folder):
        if not subfolder.startswith('data_'): continue
        subfolder = os.path.join(parent_folder, subfolder)
        for subsubfolder in os.listdir(subfolder):
            subsubfolder = os.path.join(subfolder, subsubfolder)
            for sub3folder in os.listdir(subsubfolder):
                if not sub3folder.startswith('CASHFE-'): continue
                sub4folder_path = os.path.join(subsubfolder, sub3folder)
                if os.path.isdir(sub4folder_path):
                    task_id, task_type, has_filter_params = get_task_info(sub4folder_path)
                    if task_id is not None and task_type is not None:
                        topk_paths = get_topk_config_paths(sub4folder_path)
                        for path in topk_paths:
                            task_data.append((task_type, has_filter_params, task_id, path))

    # Sort the data
    task_data.sort(key=lambda x: (x[0], not x[1], cls_rank.index(x[2]) if x[0]==0 else rgs_rank.index(x[2])))

    res_dict = OrderedDict()
    # Output the sorted paths
    for a, b, c, path in task_data:
        # print(a, b, c, path)
        if a not in res_dict:
            res_dict[a] = OrderedDict()
        if b not in res_dict[a]:
            res_dict[a][b] = []
        res_dict[a][b].append((c, path))

    # for key, value in res_dict.items():
    #     for subkey, subvalue in value.items():
    #         print(key, subkey)
    #         for sv in subvalue:
    #             print('"%s" \\' % sv)

    breakpoint()
    return res_dict


if __name__ == "__main__":
    # parent_folder = '/root/mindware/examples/benchmark/refit_data'
    parent_folder = '/root/mindware/examples/benchmark/benchmark_data'
    main(parent_folder)
