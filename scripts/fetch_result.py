import pickle as pkl
import numpy as np
import os

task = 'cls'

if task == 'cls':
    dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_embedding.pkl', 'rb'))
    meta_res_dir = './data_cls/meta_res/'
    algo2perf_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_algo2perf.pkl'
    algorithms = [ 'adaboost', 'extra_trees', 'gradient_boosting', 'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'lightgbm', 'logistic_regression', 'qda', 'random_forest', 'xgboost' ]
    metrics = ['acc', 'f1', 'auc']
else:
    
    dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_embedding.pkl', 'rb'))
    meta_res_dir = './data_rgs/meta_res/'
    algo2perf_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_algo2perf.pkl'
    algorithms = [ 'adaboost', 'extra_trees', 'gradient_boosting', 'k_nearest_neighbors', 'lasso_regression', 'liblinear_svr', 'libsvm_svr', 'lightgbm', 'random_forest', 'ridge_regression', 'xgboost' ]
    metrics = ['mse', 'r2', 'mae']



dataset_ids = dataset_embedding['task_ids']
dataset_ids = [tmp[5:] for tmp in dataset_ids]

algo4perf_dict = {}

for metric in metrics:
    algo4perf = np.full((len(dataset_ids), len(algorithms)), np.nan)
    algo4perf_dict[metric] = algo4perf

    for i, data in enumerate(dataset_ids):
        for j, algo in enumerate(algorithms):
            scores = []
            for rep in range(3):
                path = os.path.join(meta_res_dir, "%s-%s-%s-%d-1200.pkl" % (data, algo, metric, rep))
                if not os.path.exists(path):
                    print(i, j, rep, path, "Not exist!")
                    continue
                res = pkl.load(open(path, 'rb'))
                scores.append(res[3])

            if len(scores) >= 1:
                algo4perf[i, j] = np.median(scores)
            else:
                algo4perf[i, j] = -np.inf


algo2perf = {
    'task_ids': dataset_embedding['task_ids'],
    'algorithms_included': algorithms,
    'perf4algo': algo4perf_dict
}
with open(algo2perf_path, 'wb') as f:
    pkl.dump(algo2perf, f)