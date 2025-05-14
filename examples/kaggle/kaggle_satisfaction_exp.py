import pandas as pd
import os
import sys


# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mindware.utils.data_manager import DataManager
from mindware import REGRESSION, CLASSIFICATION
import argparse
import subprocess
import zipfile
from mindware import CASH, CASHFE, FE, HPO, ENS
import multiprocessing as mp
import numpy as np

include_algorithms = [
    'extra_trees', 'gradient_boosting',
    'random_forest', 'lightgbm', 'xgboost'
]

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle_id', type=str, default="santander-customer-satisfaction")

    parser.add_argument('--ensemble_method', type=str, default="ensemble_selection", help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--metric', type=str, default='auc', help='metric')
    parser.add_argument('--thread', type=int, default=18, help='thread')

    parser.add_argument('--layer_upper', type=int, default=4)
    parser.add_argument('--size_upper', type=int, default=30)


    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=21600, help='time limit')
    parser.add_argument('--per_run_time_limit', type=int, default=600, help='time limit')
    parser.add_argument('--inner_iter_num_per_iter', type=int, default=10)

    parser.add_argument('--n_algorithm', type=int, default=-1)
    parser.add_argument('--n_preprocessor', type=int, default=-1)

    parser.add_argument('--refit', type=str, choices=['partial', 'full', 'cv'], default='full')
    parser.add_argument('--output_dir', type=str, default='./data')
    args = parser.parse_args()

    task_type = CLASSIFICATION
    kaggle_id = args.kaggle_id
    id_name = 'ID'
    res_name = 'TARGET'

    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", kaggle_id],
        capture_output=True,
        text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    with zipfile.ZipFile(f"{kaggle_id}.zip", "r") as zip_ref:
        zip_ref.extractall(kaggle_id)

    dm = DataManager()
    train_csv = dm.load_csv(os.path.join(kaggle_id, 'train.csv'))
    train_data_node = dm.from_train_df(train_csv, drop_index=[0], label_col=-1)
    train_data_node = dm.preprocess_fit(train_data_node, task_type)

    test_csv = dm.load_csv(os.path.join(kaggle_id, 'test.csv'))
    test_data_node = dm.from_test_df(test_csv)
    test_data_node = dm.preprocess_transform(test_data_node)

    id_data = pd.read_csv(os.path.join(kaggle_id, 'test.csv'))[id_name]

    # filter_params = {}
    # if args.n_algorithm != -1:
    #     include_algorithms = None
    #     filter_params['n_algorithm'] = args.n_algorithm
    # if args.n_preprocessor != -1:
    #     filter_params['n_preprocessor'] = args.n_preprocessor
    
    # opt = CASHFE(
    #     include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
    #     metric=args.metric,
    #     data_node=train_data_node, evaluation=args.evaluation,
    #     optimizer="block_1", inner_iter_num_per_iter=args.inner_iter_num_per_iter,
    #     time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=args.per_run_time_limit,
    #     output_dir=args.output_dir, seed=1, n_jobs=1,
    #     ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size, task_id=args.kaggle_id,
    #     filter_params = filter_params,
    # )

    # print(opt.get_conf(save=True))  # 保存设置
    # _, stats_path = opt.run()
    # print(opt.get_model_info(save=True))  # 保存最优模型信息
    # pred_hpo = opt.predict_proba(test_data_node, ens=False)[:, 1]

    # result_hpo = pd.DataFrame({id_name: id_data, res_name: pred_hpo})
    # best_path = os.path.join(kaggle_id, f'best_{args.time_limit}_result.csv')
    # result_hpo.to_csv(best_path, index=False)
    # print('Result has been saved to result_hpo.csv.')
    
    # pred_ens = opt.predict_proba(test_data_node, ens=True)[:, 1]
    
    # result_ens = pd.DataFrame({id_name: id_data, res_name: pred_ens})
    # result_ens.to_csv(os.path.join(kaggle_id, f'ens_{args.time_limit}_ens{args.ensemble_method}{args.ensemble_size}_result.csv'), index=False)
    # print('Ensemble result has been saved to result_ens.csv.')

    stats_path = '/root/xbc/mindware/examples/kaggle/data_comp/CASHFE-block_1(1)-holdout_santander-customer-satisfaction_2025-05-01-18-34-37-633817/2025-05-01-18-37-15-223802_topk_config.pkl'

    import pickle as pkl
    with open(stats_path, 'rb') as f:
        stats = pkl.load(f)
    dir_name = os.path.dirname(stats_path)
    for key in stats.keys():
        for i in range(len(stats[key])):
            tmp = stats[key][i]
            stats[key][i] = (tmp[0], tmp[1], os.path.join(dir_name, os.path.basename(tmp[2])))

    opt = ENS(task_type=task_type, stats=stats,
                metric=args.metric, data_node=train_data_node,
                optimizer='smac',
                time_limit=1, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=kaggle_id, 
                layer_upper=args.layer_upper, size_upper=args.size_upper)  # 默认配置
    
    opt.run(refit=args.refit)
    opt.get_model_info(save=True)  # 保存最优模型信息
    ens_preds = opt.predict_proba(test_data_node, refit=args.refit)
    topk = len(ens_preds) // 2
    for i, mode in enumerate(['top', 'only']):
        for k in range(topk):
            
            pred_ens = ens_preds[i*topk+k][:, 1]
            result_ens = pd.DataFrame({id_name: id_data, res_name: pred_ens})
            result_ens.to_csv(os.path.join(kaggle_id, f'ens_def_{mode}{topk-k}_result.csv'), index=False)

    # opt = ENS(task_type=task_type, stats=stats,
    #             metric=args.metric, data_node=train_data_node,
    #             optimizer='smac',
    #             time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
    #             output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=kaggle_id, 
    #             layer_upper=args.layer_upper, size_upper=args.size_upper)
    # print(opt.get_conf(save=True))
    # opt.run(refit=args.refit)
    # opt.get_model_info(save=True)  # 保存最优模型信息
    # ens_preds = opt.predict_proba(test_data_node, refit=args.refit)

    # topk = len(ens_preds) // 2
    # for i, mode in enumerate(['top', 'only']):
    #     for k in range(topk):
            
    #         pred_ens = ens_preds[i*topk+k][:, 1]
    #         result_ens = pd.DataFrame({id_name: id_data, res_name: pred_ens})
    #         result_ens.to_csv(os.path.join(kaggle_id, f'ens_{args.time_limit}opt_{mode}{topk-k}_result.csv'), index=False)

