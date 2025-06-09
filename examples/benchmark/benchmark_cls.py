import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import argparse
import numpy as np
import json
from mindware.utils.data_manager import DataManager
from mindware.components.utils.constants import *
from mindware import CASH, CASHFE, FE, HPO, ENS
import multiprocessing as mp
from mindware.components.metrics.metric import get_metric
from benchmark_utils import get_dataset_info

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS     # export NUMEXPR_NUM_THREADS=1


include_algorithms = [
    'adaboost', 'extra_trees', 'gradient_boosting',
    'random_forest', 'lightgbm', 'xgboost'
]

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # 从命令行参数中解析出参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_seed', type=int, default=1)
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--estimator_id', type=str, default='xgboost', help='for fe and hpo')
    parser.add_argument('--optimizer', type=str, default='block_1', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default=None, help='normalize, minmax')

    parser.add_argument('--ensemble_method', type=str, default='ensemble_selection', help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--ratio', type=float, default=0.4, help='ensemble size')
    parser.add_argument('--layer', type=int, default=0, help='ensemble threads')
    parser.add_argument('--thread', type=int, default=20, help='ensemble threads')

    parser.add_argument('--layer_upper', type=int, default=4)
    parser.add_argument('--size_upper', type=int, default=40)

    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=3600, help='time limit')
    parser.add_argument('--per_time_limit', type=int, default=300, help='time limit')
    parser.add_argument('--inner_iter_num_per_iter', type=int, default=10)

    parser.add_argument('--n_algorithm', type=int, default=-1)
    parser.add_argument('--n_preprocessor', type=int, default=-1)

    parser.add_argument('--refit', type=str, choices=['partial', 'full', 'cv'], default='full')
    parser.add_argument('--stats_path', type=str, default=None)

    parser.add_argument('--job_id', type=str, nargs='*', help='job index')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--output_file', type=str, default='results.txt')
    args = parser.parse_args()

    task_type = CLASSIFICATION
    metric = 'acc'
    estimator_id = args.estimator_id

    if args.Opt == 'cash':
        # 'lda',
        OPT = CASH
    elif args.Opt == 'cashfe':
        OPT = CASHFE
    elif args.Opt == 'fe':
        OPT = FE
    elif args.Opt == 'hpo':
        OPT = HPO
    elif args.Opt == 'ens':
        OPT = ENS
    else:
        raise ValueError("Not supprt Opt type:", args.Opt)

    for dataset in args.job_id:
        dataset_path, label_column, header, sep = get_dataset_info('CLS', dataset)
        dm = DataManager()
        df = dm.load_csv(dataset_path, header=header, sep=sep)
        train_df, test_df = dm.split_data(df, label_col=label_column,
                                          test_size=0.2, random_state=args.train_test_split_seed, task_type=task_type)
        train_data_node = dm.from_train_df(train_df, label_col=label_column)
        train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=args.x_encode)

        test_data_node = dm.from_test_df(test_df, has_label=True)
        test_data_node = dm.preprocess_transform(test_data_node)

        scorer = get_metric(metric)
        if args.stats_path is None:

            inc_alg = include_algorithms
            if dataset in ['covertype', 'higgs', 'mv']:
                inc_alg = [alg for alg in include_algorithms if alg not in ['adaboost']]

            filter_params = {}
            if args.n_algorithm != -1:
                inc_alg = None
                filter_params['n_algorithm'] = args.n_algorithm
            if args.n_preprocessor != -1:
                filter_params['n_preprocessor'] = args.n_preprocessor

            per_run_time_limit = 300
            if args.ensemble_method == 'cv':
                per_run_time_limit *= 2

            import pickle as pkl
            stats_paths = pkl.load(open(os.path.join(args.output_dir, './stats_path.pkl'), 'rb'))
            stats_path = None
            for tmp in stats_paths[0][True]:
                if tmp[0] == dataset:
                    stats_path = tmp[1]
                    break

            if stats_path is None:
                if args.Opt in ['cash', 'cashfe']:
                    opt = OPT(
                        include_algorithms=inc_alg, sub_optimizer='smac', task_type=task_type,
                        metric=metric,
                        data_node=train_data_node, evaluation=args.evaluation, resampling_params={'folds': 5, 'ratio': args.ratio},
                        optimizer=args.optimizer, inner_iter_num_per_iter=args.inner_iter_num_per_iter,
                        time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=per_run_time_limit,
                        output_dir=args.output_dir, seed=1, n_jobs=1,
                        ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size, task_id=dataset,
                        filter_params=filter_params
                    )
                elif args.Opt in ['fe', 'hpo']:
                    opt = OPT(
                        estimator_id=estimator_id, task_type=task_type,
                        metric=metric,
                        data_node=train_data_node, evaluation=args.evaluation, resampling_params=None,
                        optimizer=args.optimizer,
                        time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=300,
                        output_dir=args.output_dir, seed=1, n_jobs=1,
                        ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size, task_id=dataset,
                    )

                print(opt.get_conf(save=True))
                _, stats_path = opt.run(refit=args.refit)
                opt.get_model_info(save=True)  # 保存最优模型信息
                pred = opt.predict(test_data_node, refit=args.refit, ens=False)
                perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign

                ens_perf = None
                if args.ensemble_method:
                    ens_pred = opt.predict(test_data_node, refit=args.refit, ens=True)
                    ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign

            with open(stats_path, 'rb') as f:
                stats = pkl.load(f)
            dir_name = os.path.dirname(stats_path)
            for key in stats.keys():
                for i in range(len(stats[key])):
                    tmp = stats[key][i]
                    stats[key][i] = (tmp[0], tmp[1], os.path.join(dir_name, os.path.basename(tmp[2])))

            tar_dir = os.path.join(args.output_dir, 'results_cls')
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            tar_file = os.path.join(tar_dir, f'{dataset}.json')
            if os.path.exists(tar_file):
                with open(tar_file, 'r') as f:
                    res_dict = json.load(f)
            else:
                res_dict = {
                    'task_type': 'CLS',
                    'dataset': dataset,
                    'refit': args.refit,
                    'filter': f'm{args.n_algorithm}_p{args.n_preprocessor}',
                    'time_limit': args.time_limit,
                }


            if 'best' not in res_dict:
                pred = OPT._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats,
                                        refit=args.refit, output_dir=args.output_dir, task_id=dataset)
                perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign
                res_dict['best'] = perf

            if 'fixed_ens' not in res_dict:
                ens_perf = None
                if args.ensemble_method:
                    ratio = f'_{args.ratio}' if args.ensemble_method in ['blending', 'stacking'] else ''
                    layer = f'_L{args.layer+1}' if args.ensemble_method in ['blending', 'stacking'] else ''
                    ens_str = f'{args.ensemble_method}{args.ensemble_size}{ratio}{layer}' if args.ensemble_method is not None else 'none'
                    ens_pred = OPT._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats,
                                                resampling_params={'folds': 5, 'stack_layers': args.layer},
                                                ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size,
                                                refit=args.refit, output_dir=args.output_dir, task_id=dataset)
                    ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign
                    res_dict['fixed_ens'] = {ens_str: ens_perf}

            defopt_ens_dict = {}
            if 'defopt_ens' in res_dict:
                defopt_ens_dict = res_dict['defopt_ens']

            for size_def, ratio_def, dropout_def in [
                (10, 20, 20), (10, 40, 20), (20, 20, 20), (20, 40, 20),
                (10, 20, 0), (10, 40, 0), (20, 20, 0), (20, 40, 0)
                ]:
                key = f's{size_def}_r{ratio_def}_d{dropout_def}'
                if key not in defopt_ens_dict:
                    opt = ENS(task_type=task_type, stats=stats,
                                metric=metric, data_node=train_data_node,
                                optimizer='smac',
                                time_limit=1, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                                output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset, 
                                val_nodes={'test': test_data_node},
                                layer_upper=args.layer_upper, size_upper=args.size_upper,
                                size_default=size_def, ratio_default=ratio_def, dropout_default=dropout_def)

                    print(opt.get_conf(save=True))
                    opt.run(refit=args.refit)
                    print(opt.get_model_info(save=True))  # 保存最优模型信息
                    ens_def_preds = opt.predict(test_data_node, refit=args.refit)
                    ens_def_perfs = []
                    for ens_def_pred in ens_def_preds:
                        ens_def_perf = scorer._score_func(test_data_node.data[1], ens_def_pred) * scorer._sign
                        ens_def_perfs.append(str(ens_def_perf))
                    ens_def_perf = ', '.join(ens_def_perfs)
                    defopt_ens_dict[key] = ens_def_perf
            res_dict['defopt_ens'] = defopt_ens_dict

            if 'opt_ens' not in res_dict:
                opt = ENS(task_type=task_type, stats=stats,
                            metric=metric, data_node=train_data_node,
                            optimizer='smac',
                            time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                            output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
                            val_nodes={'test': test_data_node},
                            layer_upper=args.layer_upper, size_upper=args.size_upper)

                print(opt.get_conf(save=True))
                opt.run(refit=args.refit)
                print(opt.get_model_info(save=True))  # 保存最优模型信息
                ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
                ens_opt_perfs = []
                for ens_opt_pred in ens_opt_preds:
                    ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
                    ens_opt_perfs.append(str(ens_opt_perf))

                res_dict['opt_ens'] = ens_opt_perfs

            if 'opt_ens_d0' not in res_dict:
                opt = ENS(task_type=task_type, stats=stats,
                            metric=metric, data_node=train_data_node,
                            optimizer='smac',
                            time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                            output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
                            val_nodes={'test': test_data_node},
                            layer_upper=args.layer_upper, size_upper=args.size_upper, dropout_default=0)

                print(opt.get_conf(save=True))
                opt.run(refit=args.refit)
                print(opt.get_model_info(save=True))  # 保存最优模型信息
                ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
                ens_opt_perfs = []
                for ens_opt_pred in ens_opt_preds:
                    ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
                    ens_opt_perfs.append(str(ens_opt_perf))

                res_dict['opt_ens_d0'] = ens_opt_perfs

            if 'opt_ens_d0-20' not in res_dict:
                opt = ENS(task_type=task_type, stats=stats,
                            metric=metric, data_node=train_data_node,
                            optimizer='smac',
                            time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                            output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
                            val_nodes={'test': test_data_node},
                            layer_upper=args.layer_upper, size_upper=args.size_upper, dropout_default=0)

                print(opt.get_conf(save=True))
                opt.run(refit=args.refit)
                print(opt.get_model_info(save=True))  # 保存最优模型信息
                ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
                ens_opt_perfs = []
                for ens_opt_pred in ens_opt_preds:
                    ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
                    ens_opt_perfs.append(str(ens_opt_perf))

                res_dict['opt_ens_d0-20'] = ens_opt_perfs

            # if 'opt_ens_train' not in res_dict:
            #     opt = ENS(task_type=task_type, stats=stats,
            #                 metric=metric, data_node=train_data_node,
            #                 optimizer='smac',
            #                 time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
            #                 output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #                 val_nodes={'test': test_data_node}, judge = 'train',
            #                 layer_upper=args.layer_upper, size_upper=args.size_upper, dropout_default=20)

            #     print(opt.get_conf(save=True))
            #     opt.run(refit=args.refit)
            #     print(opt.get_model_info(save=True))  # 保存最优模型信息
            #     ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
            #     ens_opt_perfs = []
            #     for ens_opt_pred in ens_opt_preds:
            #         ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
            #         ens_opt_perfs.append(str(ens_opt_perf))

            #     res_dict['opt_ens_train'] = ens_opt_perfs

            with open(tar_file, 'w') as f:
                json.dump(res_dict, f, indent=4)

        else:
            import pickle as pkl
            with open(args.stats_path, 'rb') as f:
                stats = pkl.load(f)
            dir_name = os.path.dirname(args.stats_path)
            for key in stats.keys():
                for i in range(len(stats[key])):
                    tmp = stats[key][i]
                    stats[key][i] = (tmp[0], tmp[1], os.path.join(dir_name, os.path.basename(tmp[2])))

            if args.Opt == 'ens':

                pred = CASHFE._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats, 
                                        refit=args.refit, output_dir=args.output_dir, task_id=dataset)
                perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign

                opt = OPT(task_type=task_type, stats=stats,
                          metric=metric, data_node=train_data_node,
                          optimizer='smac',
                          time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
                          output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset, 
                          val_nodes={'test': test_data_node},
                          layer_upper=args.layer_upper, size_upper=args.size_upper)

                print(opt.get_conf(save=True))
                opt.run(refit=args.refit)
                print(opt.get_model_info(save=True))  # 保存最优模型信息
                ens_preds = opt.predict(test_data_node, refit=args.refit)
                ens_perfs = []
                for ens_pred in ens_preds:
                    ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign
                    ens_perfs.append(str(ens_perf))

                ens_perf = ', '.join(ens_perfs)

            else:

                pred = OPT._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats,
                                        refit=args.refit, output_dir=args.output_dir, task_id=dataset)
                perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign

                ens_perf = None
                if args.ensemble_method:
                    ens_pred = OPT._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats,
                                                resampling_params={'folds': 5, 'stack_layers': args.layer},
                                                ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size,
                                                refit=args.refit, output_dir=args.output_dir, task_id=dataset)
                    ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign

            with open(args.output_file, 'a+') as f:
                if args.Opt == 'ens':
                    ens_str = 'ensopt'
                else:
                    ratio = f'_{args.ratio}' if args.ensemble_method in ['blending', 'stacking'] else ''
                    layer = f'_L{args.layer+1}' if args.ensemble_method in ['blending', 'stacking'] else ''
                    ens_str = f'{args.ensemble_method}{args.ensemble_size}{ratio}{layer}' if args.ensemble_method is not None else 'none'

                f.write(f'CLS: {args.Opt}-{args.optimizer}-{args.refit}-{ens_str}-filter_m{args.n_algorithm}_p{args.n_preprocessor}, {dataset}: {perf}, {ens_perf}\n')

