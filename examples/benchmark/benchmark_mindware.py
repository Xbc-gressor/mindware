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
from mindware.components.ensemble.unnamed_ensemble import choose_base_models_classification, \
    choose_base_models_regression

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
    parser.add_argument('--task_type', type=str)
    parser.add_argument('--train_test_split_seed', type=int, default=1)
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--estimator_id', type=str, default='xgboost', help='for fe and hpo')
    parser.add_argument('--optimizer', type=str, default='block_1', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default=None, help='normalize, minmax')

    parser.add_argument('--ensemble_method', type=str, default=None, help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--ratio', type=float, default=0.4, help='ensemble size')
    parser.add_argument('--meta_learner', type=str, default='weighted', help='ensemble size')
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

    parser.add_argument('--epoch', type=int, default='1000')
    parser.add_argument('--mode', type=str, default='model_averaging')
    parser.add_argument('--refit', type=str, choices=['partial', 'full', 'cv'], default='full')
    parser.add_argument('--stats_path', type=str, default=None)

    parser.add_argument('--job_id', type=str, nargs='*', help='job index')
    parser.add_argument('--output_dir', type=str, default='./benchmark_data')
    parser.add_argument('--output_file', type=str, default='results.txt')
    args = parser.parse_args()

    task_str = args.task_type
    task_type = CLASSIFICATION
    metric = 'acc'
    if task_str == 'RGS':
        task_type = REGRESSION
        metric = 'mse'
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
        dataset_path, label_column, header, sep = get_dataset_info(task_str, dataset)
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
            if args.evaluation == 'cv':
                per_run_time_limit *= 2

            stats_path = None
            tmp_path = os.path.join(args.output_dir, './stats_path.pkl')
            if os.path.exists(tmp_path):
                import pickle as pkl
                stats_paths = pkl.load(open(tmp_path, 'rb'))
                for tmp in stats_paths[task_type][True].get(args.time_limit, []):
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
                        ensemble_method=None, ensemble_size=args.ensemble_size, task_id=dataset,
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
                        ensemble_method=None, ensemble_size=args.ensemble_size, task_id=dataset,
                    )

                print(opt.get_conf(save=True))
                _, stats_path = opt.run(refit=args.refit)
                opt.get_model_info(save=True)  # 保存最优模型信息
                # pred = opt.predict(test_data_node, refit=args.refit, ens=False)
                # perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign

                # ens_perf = None
                # if args.ensemble_method:
                #     ens_pred = opt.predict(test_data_node, refit=args.refit, ens=True)
                #     ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign

            with open(stats_path, 'rb') as f:
                stats = pkl.load(f)
            dir_name = os.path.dirname(stats_path)
            for key in stats.keys():
                for i in range(len(stats[key])):
                    tmp = stats[key][i]
                    stats[key][i] = (tmp[0], tmp[1], os.path.join(dir_name, os.path.basename(tmp[2])))

            tar_dir = os.path.join(args.output_dir, f'results_{task_str}_{args.time_limit}')
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            tar_file = os.path.join(tar_dir, f'{dataset}.json')
            if os.path.exists(tar_file):
                with open(tar_file, 'r') as f:
                    res_dict = json.load(f)
            else:
                res_dict = {
                    'task_type': task_str,
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

            # if 'dropoutreg_exp' not in res_dict:
            #     res_dict['dropoutreg_exp'] = {}
            # import datetime
            # from mindware.modules.base_evaluator import BaseEvaluator
            # from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
            # from mindware.utils.functions import is_imbalanced_dataset
            # from copy import deepcopy
            # import time
            # import shutil
            # from mindware.utils.logging_utils import setup_logger

            # _path = 'dropout-(%s)-%s_%s' % (
            #     task_str, dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
            # )
            # output_dir = os.path.join(args.output_dir, _path)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # logger_name = 'MindWare-dropout-(%d)' % (1)
            # setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))

            # if_imbal = False
            # if task_type in CLS_TASKS:
            #     if_imbal = is_imbalanced_dataset(train_data_node)
            # train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=train_data_node, seed=1)
            # val_nodes = {'test': test_data_node, 'val': valid_node}

            # n_dim = 1
            # if task_type in CLS_TASKS:
            #     unique_num = len(np.unique(train_data_node.data[1]))
            #     if unique_num > 2:
            #         n_dim = unique_num

            # es = EnsembleBuilder(stats=stats, valid_node=valid_node,
            #         task_type=task_type, if_imbal=if_imbal,
            #         metric=scorer,
            #         output_dir=output_dir, seed=1, thread=args.thread)
            # for ens_size in [30]:
            #     for ratio in [0.3]:

            #         base_features_backup, ori_xs = None, None
            #         for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            #             ratio_str = f'_{ratio}'
            #             layer = f'_L1'
            #             meta = f'_{args.meta_learner}'
            #             ens_str = f'stacking{ens_size}{ratio_str}_L1{meta}dropout{dropout}'

            #             if ens_str not in res_dict['dropoutreg_exp']:
            #                 es.build_ensemble(
            #                     ensemble_method='stacking', ensemble_size=ens_size, ratio=ratio, judge='val', opt=True,
            #                     stack_layers=0, meta_learner=args.meta_learner, val_nodes=val_nodes
            #                 )
            #                 model = es.model
            #                 if len(train_node.data[1].shape) == 1 and model.task_type in CLS_TASKS:
            #                     reshape_y = np.reshape(train_node.data[1], (len(train_node.data[1]), 1))
            #                     model.encoder.fit(reshape_y)
            #                 # Train basic models using a part of training data
            #                 if base_features_backup is None:
            #                     _mode = 'partial' if model.judge == 'val' else 'full'
            #                     base_features_backup, ori_xs = model.get_base_features(train_node, val_nodes, mode=_mode)
            #                 base_features = deepcopy(base_features_backup)

            #                 if dropout > 0:
            #                     dropout_num = int(ens_size * dropout)
            #                     if dropout_num > 0:
            #                         rng = np.random.default_rng(seed=1)

            #                         train_num, all_dim = base_features['train'].shape
            #                         predict_dim = base_features['train'].shape[1]
            #                         ori_dim = all_dim - predict_dim
            #                         n_dim = predict_dim // ens_size
            #                         dropout_mask = np.zeros((train_num, ens_size), dtype=int)
            #                         for i in range(train_num):
            #                             dropout_mask[i, rng.choice(ens_size, dropout_num, replace=False)] = 1

            #                         for dim in range(n_dim):
            #                             col =[ori_dim + dim + idx * n_dim for idx in range(ens_size)]
            #                             data_pure = base_features['train'][:, col] * (1 - dropout_mask)
            #                             base_features['train'][:, col] = data_pure + dropout_mask * np.sum(data_pure, axis=1, keepdims=True) / (ens_size - dropout_num)

            #                 final_labels = {'train': train_node.data[1]}
            #                 if val_nodes is not None:
            #                     for key in val_nodes.keys():
            #                         final_labels[key] = val_nodes[key].data[1]

            #                 model.forward(base_features, final_labels, train=True, ori_xs=ori_xs)
            #                 model_info = es.get_ens_model_info()
            #                 res_dict['dropoutreg_exp'][ens_str] = deepcopy((model.best_stack.meta_learners[0].coef_.tolist(), model_info['leader_board']))

            # shutil.rmtree(output_dir)


            # if 'diversityavg_exp' not in res_dict:
            #     res_dict['diversityavg_exp'] = {}
            # import datetime
            # from mindware.modules.base_evaluator import BaseEvaluator
            # from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
            # from mindware.utils.functions import is_imbalanced_dataset
            # from copy import deepcopy
            # import time
            # import shutil
            # from mindware.utils.logging_utils import setup_logger

            # _path = 'diverty-(%s)-%s_%s' % (
            #     task_str, dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
            # )
            # output_dir = os.path.join(args.output_dir, _path)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # logger_name = 'MindWare-dropout-(%d)' % (1)
            # setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))

            # if_imbal = False
            # if task_type in CLS_TASKS:
            #     if_imbal = is_imbalanced_dataset(train_data_node)
            # train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=train_data_node, seed=1)
            # val_nodes = {'test': test_data_node, 'val': valid_node}

            # n_dim = 1
            # if task_type in CLS_TASKS:
            #     unique_num = len(np.unique(train_data_node.data[1]))
            #     if unique_num > 2:
            #         n_dim = unique_num

            # es = EnsembleBuilder(stats=stats, valid_node=valid_node,
            #         task_type=task_type, if_imbal=if_imbal,
            #         metric=scorer,
            #         output_dir=output_dir, seed=1, thread=args.thread)

            # model_cnt = 0
            # perfs = []
            # for algo_id in es.stats.keys():
            #     model_to_eval = es.stats[algo_id]
            #     for idx, (config, perf, path) in enumerate(model_to_eval):
            #         perfs.append(perf)
            #         model_cnt += 1
            # perfs = np.array(perfs)
            # for ens_size in [30]:
            #     if len(es.predictions) < ens_size:
            #         exit(1)
            #     for ratio in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]:
            #         ratio_str = f'_{ratio}'
            #         ens_str = f'stacking{ens_size}{ratio_str}'

            #         if ens_str not in res_dict['diversityavg_exp']:

            #             y_valid = es.valid_node.data[1]
            #             if task_type in CLS_TASKS:
            #                 base_model_mask, sel_G = choose_base_models_classification(
            #                     np.array(es.predictions), np.array(y_valid), ens_size, ratio=ratio
            #                 )
            #             else:
            #                 base_model_mask, sel_G = choose_base_models_regression(
            #                     np.array(es.predictions), np.array(y_valid), ens_size, ratio=ratio
            #                 )
            #             res_dict['diversityavg_exp'][ens_str] = [perfs[np.where(base_model_mask)[0]].sum(), np.diag(sel_G).sum(), sel_G.sum() - np.diag(sel_G).sum()]

            # shutil.rmtree(output_dir)


            # if 'struc_exp' not in res_dict:
            #     res_dict['struc_exp'] = {}
            # import datetime
            # from mindware.modules.base_evaluator import BaseEvaluator
            # from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
            # from mindware.utils.functions import is_imbalanced_dataset
            # from copy import deepcopy
            # import time
            # import shutil
            # from mindware.utils.logging_utils import setup_logger

            # _path = 'Struc-(%s)-%s_%s' % (
            #     task_str, dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
            # )
            # output_dir = os.path.join(args.output_dir, _path)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # logger_name = 'MindWare-struc-(%d)' % (1)
            # setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))

            # if_imbal = False
            # if task_type in CLS_TASKS:
            #     if_imbal = is_imbalanced_dataset(train_data_node)
            # train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=train_data_node, seed=1)
            # val_nodes = {'test': test_data_node, 'val': valid_node}

            # es = EnsembleBuilder(stats=stats, valid_node=valid_node,
            #         task_type=task_type, if_imbal=if_imbal,
            #         metric=scorer,
            #         output_dir=output_dir, seed=1, thread=args.thread)

            # model_cnt = 0
            # perfs = []
            # for algo_id in es.stats.keys():
            #     model_to_eval = es.stats[algo_id]
            #     for idx, (config, perf, path) in enumerate(model_to_eval):
            #         perfs.append(perf)
            #         model_cnt += 1
            # perfs = np.array(perfs)
            # for ens_size in [30]:
            #     for ratio in [0.3]:
            #         for dropout in [0, 0.1, 0.2, 0.3, 0.4]:
            #             for retain in [False, True]:
            #                 if retain != True and dropout != 0.2:
            #                     continue
            #                 if retain == True and dropout == 0.2:
            #                     layer = 4
            #                 else:
            #                     layer = 2
            #                 ratio_str = f'_{ratio}'
            #                 ens_str = f'stacking{ens_size}{ratio_str}_dropout{dropout}_retain{retain}'

            #                 # if dropout != 0 or ens_str not in res_dict['struc_exp']:
            #                 es.build_ensemble(
            #                     ensemble_method='stacking', ensemble_size=ens_size, ratio=ratio, judge='val', opt=True,
            #                     stack_layers=layer, meta_learner='auto', val_nodes=val_nodes, retain=retain, dropout=dropout
            #                 )
            #                 es.fit(datanode=train_node, val_nodes=val_nodes)
            #                 model_info = es.get_ens_model_info()
            #                 res_dict['struc_exp'][ens_str] = deepcopy(model_info['leader_board'])

            # shutil.rmtree(output_dir)

            # if 'retain_exp' not in res_dict:
            #     res_dict['retain_exp'] = {}
            # import datetime
            # from mindware.modules.base_evaluator import BaseEvaluator
            # from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
            # from mindware.utils.functions import is_imbalanced_dataset
            # from copy import deepcopy
            # import time
            # import shutil
            # from mindware.utils.logging_utils import setup_logger

            # _path = 'Retain-(%s)-%s_%s' % (
            #     task_str, dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
            # )
            # output_dir = os.path.join(args.output_dir, _path)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # logger_name = 'MindWare-Retain-(%d)' % (1)
            # setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))

            # if_imbal = False
            # if task_type in CLS_TASKS:
            #     if_imbal = is_imbalanced_dataset(train_data_node)
            # train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=train_data_node, seed=1)
            # val_nodes = {'test': test_data_node, 'val': valid_node}

            # es = EnsembleBuilder(stats=stats, valid_node=valid_node,
            #         task_type=task_type, if_imbal=if_imbal,
            #         metric=scorer,
            #         output_dir=output_dir, seed=1, thread=args.thread)
            # for ens_size in [10]:
            #     for ratio in [0.4]:
            #         for retain in [False, True]:
            #             ratio_str = f'_{ratio}'
            #             layer = f'_L{args.layer+1}'
            #             meta = f'_{args.meta_learner}'
            #             ens_str = f'stacking{ens_size}{ratio_str}{layer}{meta}_retain{retain}'

            #             if ens_str not in res_dict['retain_exp']:
            #                 es.build_ensemble(
            #                     ensemble_method='stacking', ensemble_size=ens_size, ratio=ratio, judge='val', opt=True,
            #                     stack_layers=4, meta_learner='auto', val_nodes=val_nodes, retain=retain
            #                 )
            #                 es.fit(datanode=train_node, val_nodes=val_nodes)
            #                 model_info = es.get_ens_model_info()
            #                 res_dict['retain_exp'][ens_str] = deepcopy((model_info['layer_loss'], model_info['leader_board']))

            # shutil.rmtree(output_dir)

            # if 'size_exp' not in res_dict:
            #     res_dict['size_exp'] = {}
            # if args.ensemble_method:
            #     import datetime
            #     from mindware.modules.base_evaluator import BaseEvaluator
            #     from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
            #     from mindware.utils.functions import is_imbalanced_dataset
            #     from copy import deepcopy
            #     import time
            #     import shutil
            #     from mindware.utils.logging_utils import setup_logger

            #     _path = 'STA-(%s)-%s_%s' % (
            #         task_str, dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
            #     )
            #     output_dir = os.path.join(args.output_dir, _path)
            #     if not os.path.exists(output_dir):
            #         os.makedirs(output_dir)
            #     logger_name = 'MindWare-STA-(%d)' % (1)
            #     setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))

            #     if_imbal = False
            #     if task_type in CLS_TASKS:
            #         if_imbal = is_imbalanced_dataset(train_data_node)
            #     train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=train_data_node, seed=1)
            #     val_nodes = {'test': test_data_node, 'val': valid_node}

            #     es = EnsembleBuilder(stats=stats, valid_node=valid_node,
            #             task_type=task_type, if_imbal=if_imbal,
            #             metric=scorer,
            #             output_dir=output_dir, seed=1, thread=args.thread)
            #     for ens_size in [5, 10, 20, 30, 40, 50]:
            #         can_ratios = [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.49]
            #         if ens_size in [-1, 1000]:
            #             can_ratios = [0.4]
            #         for ratio in can_ratios:
            #             ratio_str = f'_{ratio}'
            #             layer = f'_L{args.layer+1}'
            #             meta = f'_{args.meta_learner}'
            #             ens_str = f'stacking{ens_size}{ratio_str}{layer}{meta}'

            #             if ens_str not in res_dict['size_exp']:
            #                 es.build_ensemble(ensemble_method='stacking', ensemble_size=ens_size, ratio=ratio, judge='val', stack_layers=args.layer, meta_learner=args.meta_learner, val_nodes=val_nodes)
            #                 es.fit(datanode=train_node, val_nodes=val_nodes)

            #                 res_dict['size_exp'][ens_str] = deepcopy(es.get_ens_model_info()['leader_board'])
            #                 with open(tar_file, 'w') as f:
            #                     json.dump(res_dict, f, indent=4)

            #     shutil.rmtree(output_dir)

            # if 'fixed_ens' not in res_dict:
            #     res_dict['fixed_ens'] = {}
            # if args.ensemble_method:
            #     # ensemble_size = args.ensemble_size
            #     # layer = args.layer
            #     for layer in [0, 1]:
            #         for ensemble_size in [1000, -1]:
            #             ratio = f'_{args.ratio}' if args.ensemble_method in ['blending', 'stacking'] else ''
            #             layer_str = f'_L{layer+1}' if args.ensemble_method in ['blending', 'stacking'] else ''
            #             meta = f'_{args.meta_learner}' if args.ensemble_method in ['blending', 'stacking'] else ''
            #             ens_str = f'{args.ensemble_method}{ensemble_size}{ratio}{layer_str}{meta}' if args.ensemble_method is not None else 'none'

            #             if ensemble_size == 1000 or ens_str not in res_dict['fixed_ens']:
            #                 ens_pred = OPT._predict_stats(task_type, metric, data_node=train_data_node, test_data=test_data_node, stats=stats,
            #                                             resampling_params={'folds': 5},
            #                                             ensemble_method=args.ensemble_method, ensemble_size=ensemble_size,
            #                                             refit=args.refit, output_dir=args.output_dir, task_id=dataset, thread=args.thread, ratio=args.ratio, stack_layers=layer, meta_learner=args.meta_learner, retain=False)
            #                 ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign
            #                 res_dict['fixed_ens'][ens_str] = ens_perf

            #                 with open(tar_file, 'w') as f:
            #                     json.dump(res_dict, f, indent=4)

            # for baseline in ['EnsOpt']:  # , 'qdo_es', f'neural_es-{args.mode}', 'cma_es'
            #     if baseline not in ['EnsOpt'] and baseline in res_dict['fixed_ens']: continue
            #     if task_str == 'RGS' and baseline == 'qdo_es': continue
            #     if baseline == 'cma_es':
            #         from mindware.modules.cma_es.mindware_cma_es import CMA_ES
            #         cma_es = CMA_ES(
            #             stats = stats, n_iterations=10000, batch_size=25, task_type=task_type, data_node = train_data_node,
            #             metric=metric, output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #             time_limit=args.time_limit
            #         )

            #         print(cma_es.get_conf(save=True))
            #         cma_es.get_weights()
            #         print(cma_es.get_model_info(save=True))  # 保存最优模型信息
            #         cma_es.refit(train_data_node, mode='full')
            #         baseline_pred = cma_es.predict(test_data_node)
            #     if baseline == 'qdo_es':
            #         from mindware.modules.qdo_es.mindware_qdo_es import QDO_ES
            #         qdo_es = QDO_ES(
            #             stats = stats, n_iterations= 10000, batch_size=20, task_type=task_type, data_node = train_data_node,
            #             metric=metric, output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #             time_limit=args.time_limit
            #         )
            #         print(qdo_es.get_conf(save=True))
            #         qdo_es.get_weights()
            #         qdo_es.refit(train_data_node, mode='full')
            #         print(qdo_es.get_model_info(save=True))  # 保存最优模型信息
            #         baseline_pred = qdo_es.predict(test_data_node)
            #     if baseline.startswith('neural_es'):
            #         from mindware.modules.neural_ensemble.base_neural_ens import Neural_ensemble
            #         neural_es = Neural_ensemble(task_type=task_type, stats=stats, mode = args.mode,
            #             data_node=train_data_node, output_dir=args.output_dir,
            #             seed=1, val_size = 0.33, metric=metric, batch_size = 128, epoch = args.epoch, n_jobs=args.thread, task_id=dataset
            #         )  # 默认配置
            #         neural_es.get_conf(save=True)  # 保存设置

            #         neural_es.run()
            #         neural_es.get_model_info(save=True)  # 保存最优模型信息
            #         baseline_pred = neural_es.predict(test_data_node)
            #     if baseline == 'OptDivBO':
            #         from mindware.modules.optdivbo.mindware_divbo import Optdivbo

            #         model = Optdivbo(iter_num = 10000 , ens_size=25, include_algorithms=include_algorithms, metric=metric,
            #                         task_type=task_type, data_node=train_data_node, test_node=test_data_node, time_limit_per_trial=300, task_id=dataset,
            #                         time_limit=args.time_limit * 2, output_dir=args.output_dir, seed=1,
            #                         filter_params=filter_params)

            #         _, baseline_pred = model.run()

            #     if baseline == 'EnsOpt':
            #         from mindware.modules.Ensopt.base_EnsOpt import BaseEnsOpt

            #         ens_opt = BaseEnsOpt(
            #             include_algorithms=inc_alg, task_type=task_type, ens_size = 12,
            #             metric=metric,
            #             data_node=train_data_node, resampling_params=None,
            #             time_limit=args.time_limit * 2, amount_of_resource=int(1e6), per_run_time_limit=300,
            #             output_dir=args.output_dir, seed=1, n_jobs=1, task_id=dataset,
            #             filter_params=filter_params
            #         )
            #         ens_opt.get_conf(save=True)  # 保存设置

            #         ens_opt.run(refit=args.refit)
            #         ens_opt.get_model_info(save=True)  # 保存最优模型信息
            #         baseline_pred = ens_opt.predict(test_data_node)

            #     baseline_perf = scorer._score_func(test_data_node.data[1], baseline_pred) * scorer._sign
            #     res_dict['fixed_ens'][baseline] = baseline_perf

            # with open(tar_file, 'w') as f:
            #     json.dump(res_dict, f, indent=4)


            # defopt_ens_dict = {}
            # if 'defopt_ens' in res_dict:
            #     defopt_ens_dict = res_dict['defopt_ens']

            # # for size_def, ratio_def, dropout_def in [
            # #     (10, 20, 20), (10, 40, 20), (20, 20, 20), (20, 40, 20),
            # #     (10, 20, 0), (10, 40, 0), (20, 20, 0), (20, 40, 0)
            # #     ]:
            # for size_def, ratio_def, dropout_def in [
            #     (10, 40, 0)
            #     ]:
            #     key = f's{size_def}_r{ratio_def}_d{dropout_def}'
            #     if key not in defopt_ens_dict:
            #         opt = ENS(task_type=task_type, stats=stats,
            #                     metric=metric, data_node=train_data_node,
            #                     optimizer='smac',
            #                     time_limit=1, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
            #                     output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #                     val_nodes={'test': test_data_node},
            #                     layer_upper=args.layer_upper, size_upper=args.size_upper,
            #                     size_default=size_def, ratio_default=ratio_def, dropout_default=dropout_def)

            #         print(opt.get_conf(save=True))
            #         opt.run(refit=args.refit)
            #         print(opt.get_model_info(save=True))  # 保存最优模型信息
            #         ens_def_preds = opt.predict(test_data_node, refit=args.refit)
            #         ens_def_perfs = []
            #         for ens_def_pred in ens_def_preds:
            #             ens_def_perf = scorer._score_func(test_data_node.data[1], ens_def_pred) * scorer._sign
            #             ens_def_perfs.append(str(ens_def_perf))
            #         ens_def_perf = ', '.join(ens_def_perfs)
            #         defopt_ens_dict[key] = ens_def_perf
            # res_dict['defopt_ens'] = defopt_ens_dict
            # with open(tar_file, 'w') as f:
            #     json.dump(res_dict, f, indent=4)

            # if 'opt_ens' not in res_dict:
            #     opt = ENS(task_type=task_type, stats=stats,
            #                 metric=metric, data_node=train_data_node,
            #                 optimizer='smac',
            #                 time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
            #                 output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #                 val_nodes={'test': test_data_node},
            #                 layer_upper=args.layer_upper, size_upper=args.size_upper)

            #     print(opt.get_conf(save=True))
            #     opt.run(refit=args.refit)
            #     print(opt.get_model_info(save=True))  # 保存最优模型信息
            #     ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
            #     ens_opt_perfs = []
            #     for ens_opt_pred in ens_opt_preds:
            #         ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
            #         ens_opt_perfs.append(str(ens_opt_perf))

            #     res_dict['opt_ens'] = ens_opt_perfs

            # if 'opt_ens_d0' not in res_dict:
            #     opt = ENS(task_type=task_type, stats=stats,
            #                 metric=metric, data_node=train_data_node,
            #                 optimizer='smac',
            #                 time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
            #                 output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #                 val_nodes={'test': test_data_node},
            #                 layer_upper=args.layer_upper, size_upper=args.size_upper, dropout_default=0)

            #     print(opt.get_conf(save=True))
            #     opt.run(refit=args.refit)
            #     print(opt.get_model_info(save=True))  # 保存最优模型信息
            #     ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
            #     ens_opt_perfs = []
            #     for ens_opt_pred in ens_opt_preds:
            #         ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
            #         ens_opt_perfs.append(str(ens_opt_perf))

            #     res_dict['opt_ens_d0'] = ens_opt_perfs

            # if 'opt_ens_d0-20' not in res_dict:
            #     opt = ENS(task_type=task_type, stats=stats,
            #                 metric=metric, data_node=train_data_node,
            #                 optimizer='smac',
            #                 time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=float(np.inf),
            #                 output_dir=args.output_dir, seed=1, n_jobs=args.thread, task_id=dataset,
            #                 val_nodes={'test': test_data_node},
            #                 layer_upper=args.layer_upper, size_upper=args.size_upper, dropout_default=0)

            #     print(opt.get_conf(save=True))
            #     opt.run(refit=args.refit)
            #     print(opt.get_model_info(save=True))  # 保存最优模型信息
            #     ens_opt_preds = opt.predict(test_data_node, refit=args.refit)
            #     ens_opt_perfs = []
            #     for ens_opt_pred in ens_opt_preds:
            #         ens_opt_perf = scorer._score_func(test_data_node.data[1], ens_opt_pred) * scorer._sign
            #         ens_opt_perfs.append(str(ens_opt_perf))

            #     res_dict['opt_ens_d0-20'] = ens_opt_perfs


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

                f.write(f'{task_str}: {args.Opt}-{args.optimizer}-{args.refit}-{ens_str}-filter_m{args.n_algorithm}_p{args.n_preprocessor}, {dataset}: {perf}, {ens_perf}\n')

