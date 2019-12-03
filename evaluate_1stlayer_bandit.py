import time
import pickle
import argparse
import numpy as np
import autosklearn.classification
from automlToolkit.datasets.utils import load_data
from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit

parser = argparse.ArgumentParser()
dataset_set = 'messidor_features,lymphography,winequality_red,winequality_white,credit,' \
                'ionosphere,splice,diabetes,pc4,spectf,spambase,amazon_employee'
parser.add_argument('--datasets', type=str, default='none')

# algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
#               'adaboost', 'random_forest', 'extra_trees', 'gradient_boosting']
algorithms = ['k_nearest_neighbors', 'libsvm_svc', 'random_forest', 'adaboost']
seed = 1
project_dir = './'


def evaluate_1stlayer_bandit(dataset='credit', trial_num=100):
    _start_time = time.time()
    raw_data = load_data(dataset, datanode_returned=True)
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data)
    bandit.optimize()
    print(bandit.final_rewards)
    print(bandit.action_sequence)
    time_cost = time.time() - _start_time

    save_path = project_dir + 'data/hierarchical_bandits_%s_%d.pkl' % (dataset, trial_num)
    with open(save_path, 'wb') as f:
        pickle.dump([bandit.final_rewards, bandit.time_records, bandit.action_sequence], f)

    return time_cost


def evaluate_autosklearn(dataset='credit', time_limit=1200):
    print('==> Start to evaluate', dataset, 'budget', time_limit)
    n_job = 1
    ensb_size = 1
    include_models = algorithms
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        include_preprocessors=None,
        exclude_preprocessors=None,
        n_jobs=n_job,
        include_estimators=include_models,
        ensemble_memory_limit=8192,
        ml_memory_limit=8192,
        ensemble_size=ensb_size,
        ensemble_nbest=ensb_size,
        initial_configurations_via_metalearning=0,
        seed=seed,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )
    print(automl)
    raw_data = load_data(dataset, datanode_returned=True)
    X, y = raw_data.data
    automl.fit(X.copy(), y.copy())
    model_desc = automl.show_models()
    print(model_desc)

    test_results = automl.cv_results_['mean_test_score']
    time_records = automl.cv_results_['mean_fit_time']
    best_result = np.max(test_results)
    print('Validation Accuracy', best_result)
    save_path = project_dir + 'data/ausk_%s.pkl' % dataset
    with open(save_path, 'wb') as f:
        pickle.dump([test_results, time_records, time_limit], f)


def benchmark(dataset):
    time_cost = evaluate_1stlayer_bandit(dataset)
    evaluate_autosklearn(dataset, int(time_cost))


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    dataset_list = list()
    if dataset_str == 'all':
        dataset_list = dataset_set
    else:
        dataset_list = dataset_str.split(',')

    for dataset in dataset_list:
        benchmark(dataset)