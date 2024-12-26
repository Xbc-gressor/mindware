from autogluon.tabular import TabularDataset, TabularPredictor
import time
import pandas as pd
import os
from datetime import datetime
import argparse
from mindware.utils.data_manager import DataManager
from mindware.components.utils.constants import *
# from sklearn.metrics._scorer import make_scorer, _BaseScorer
# from sklearn.metrics import accuracy_score

OUTPUT_FILE = 'results.txt'
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS     # export NUMEXPR_NUM_THREADS=1
task_type = REGRESSION

start_time = time.time()
start_datetime = datetime.fromtimestamp(start_time)

# 格式化为日期字符串
formatted_start_time = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=14400, help='time limit') # 5024
parser.add_argument('--job_idx', type=int, nargs='*', help='job index')
parser.add_argument('--train_test_split_seed', type=int, default=1)
args = parser.parse_args()
time_limit=args.time_limit
job_idx = args.job_idx

datasets_dir = r'/home/liuwei/automl_data/sub_automl_data/'
datasets_info = pd.read_excel(os.path.join(datasets_dir, '数据集.xlsx'), sheet_name='REG_SORT')
candidate_datasets = [
    "stock", "socmob", "Moneyball", "insurance",
    "weather_izmir", "us_crime", "debutanizer", "space_ga", "pollen", "wind", "bank8FM", "bank32nh", "kin8nm",
    "puma8NH", "cpu_act", "puma32H", "cpu_small", "visualizing_soil", "sulfur", "rainfall_bangladesh", "black_friday"
]
can_datasets_info = datasets_info[datasets_info['Datasets'].isin(candidate_datasets)].set_index('Datasets')

chosen_datasets = ['debutanizer', 'puma8NH', 'cpu_act', 'bank32nh', 'Moneyball', 'black_friday']
chosen_datasets_info = can_datasets_info.loc[chosen_datasets]
chosen_datasets_info['label_col'] = -1

if args.job_idx is not None and len(args.job_idx) > 0:
    chosen_datasets = [chosen_datasets[idx] for idx in args.job_idx]

for dataset in chosen_datasets:
    dataset_path = os.path.join(datasets_dir, 'rgs_datasets', dataset + '.csv')
    dm = DataManager()
    header = None if dataset == 'spambase' else 'infer'
    df = dm.load_csv(dataset_path, header=header)
    label_col = chosen_datasets_info.loc[dataset, 'label_col']
    train_df, test_df = dm.split_data(df, label_col=label_col,
                                        test_size=0.2, random_state=args.train_test_split_seed, task_type=task_type)
    eval_metric = 'mse'  # set this to the metric you ultimately care about
    label = train_df.columns[label_col]
    predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_df, presets=['best_quality'], time_limit=time_limit)
    perf_dct = predictor.evaluate(test_df, silent=True)
    # y_pred = predictor.predict_proba(test_df).iloc[:,1]
    # scorer = make_scorer(accuracy_score)
    # perf = scorer._score_func(test_df, y_pred) * scorer._sign

    leader_info = predictor.leaderboard(train_df)
    with open(OUTPUT_FILE, 'a+') as f:
        f.write(f'Autogluon RGS: {formatted_start_time}, {dataset}: {perf_dct}\n')
        f.write(f'leaderboard:{leader_info}\n')
    print('Result has been saved to result.txt.')



















