import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import argparse
from mindware.utils.data_manager import DataManager
from mindware.components.utils.constants import *
from mindware import CASH, CASHFE

datasets_dir = '/Users/xubeideng/Documents/icloud/Scientific Research/AutoML/sub_automl_data/'
# datasets_dir = '/root/automl_data/sub_automl_data/'

# 读取 Excel 文件中的特定 sheet
datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='CLS')
candidate_datasets = [
    "kc1", "quake", "segment", "ozone-level-8hr", "space_ga", "sick",
    "pollen", "analcatdata_supreme", "abalone", "spambase", "waveform-5000(2)",
    "phoneme", "page-blocks(2)", "optdigits", "satimage", "wind",
    "delta_ailerons", "puma8NH", "kin8nm", "puma32H", "cpu_act", "bank32nh",
    "mc1", "delta_elevators", "jm1", "pendigits", "mammography", "ailerons",
    "eeg", "letter(2)", "kropt", "mv", "fried", "2dplanes", "electricity",
    "a9a", "mnist_784", "higgs", "covertype"
]
can_datasets_info = datasets_info[datasets_info['Datasets'].isin(candidate_datasets)].set_index('Datasets')
chosen_datasets = ['kc1', 'spambase', 'cpu_act', 'ailerons', 'higgs', 'sick', 'mv', 'covertype']
chosen_datasets_info = can_datasets_info.loc[chosen_datasets]
chosen_datasets_info['label_col'] = -1
chosen_datasets_info.loc['higgs', 'label_col'] = 0
breakpoint()
"""
           Instances  Classes  Continuous  Nominal  label_col
Datasets                                                     
kc1             2109        2          21        0         -1
spambase        4600        2          57        0         -1
cpu_act         8192        2          21        0         -1
ailerons       13750        2          40        0         -1
higgs          98050        2          28        0          0
sick            3772        2           7       22         -1
mv             40768        2           7        3         -1
covertype     110393        7          14       40         -1
"""

include_algorithms = [
    'adaboost', 'extra_trees', 'gradient_boosting',
    'random_forest', 'lightgbm', 'xgboost'
]

if '__main__' == __name__:

    # 从命令行参数中解析出参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_split_seed', type=int, default=1)
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--optimizer', type=str, default='smac', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default=None, help='normalize, minmax')
    parser.add_argument('--ensemble_method', type=str, default='ensemble_selection', help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=50, help='ensemble size')
    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=3600, help='time limit')
    parser.add_argument('--per_time_limit', type=int, default=600, help='time limit')
    parser.add_argument('--job_idx', type=int, nargs='*', help='job index')
    args = parser.parse_args()

    task_type = CLASSIFICATION
    metric = 'acc'

    if args.Opt == 'cash':
        # 'lda',
        OPT = CASH
    else:
        OPT = CASHFE

    if args.job_idx is not None and len(args.job_idx) > 0:
        chosen_datasets = [chosen_datasets[idx] for idx in args.job_idx]

    for dataset in chosen_datasets:
        dataset_path = os.path.join(datasets_dir, 'cls_datasets', dataset + '.csv')
        dm = DataManager()
        header = None if dataset == 'spambase' else 'infer'
        df = dm.load_csv(dataset_path, header=header)
        train_df, test_df = dm.split_data(df, label_col=chosen_datasets_info.loc[dataset, 'label_col'],
                                          test_size=0.2, random_state=args.train_test_split_seed, task_type=task_type)
        train_data_node = dm.from_train_df(train_df, label_col=chosen_datasets_info.loc[dataset, 'label_col'])
        train_data_node = dm.preprocess_fit(train_data_node, task_type, x_encode=args.x_encode)

        test_data_node = dm.from_test_df(test_df, has_label=True)
        test_data_node = dm.preprocess_transform(test_data_node)

        opt = OPT(
            include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
            metric=metric,
            data_node=train_data_node, evaluation=args.evaluation, resampling_params=None,
            optimizer='mab', inner_iter_num_per_iter=10,
            time_limit=args.time_limit, amount_of_resource=int(1e6), per_run_time_limit=300,
            output_dir='./data', seed=1, n_jobs=1,
            ensemble_method=args.ensemble_method, ensemble_size=args.ensemble_size, task_id=dataset
        )
        print(opt.get_conf(save=True))  # 保存设置

        print(opt.run())
        print(opt.get_model_info(save=True))  # 保存最优模型信息
        scorer = opt.metric
        pred = opt.predict(test_data_node, ens=False)
        perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign

        ens_pred = opt.predict(test_data_node, ens=True)
        ens_perf = scorer._score_func(test_data_node.data[1], ens_pred) * scorer._sign

        with open('results.txt', 'a+') as f:
            f.write(f'CLS: {args.Opt}, {dataset}: {perf}, {ens_perf}\n')

        breakpoint()
