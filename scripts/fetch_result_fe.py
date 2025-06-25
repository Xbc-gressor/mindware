import pickle as pkl
import numpy as np
import pickle as pkl
import json
import os
import sys
from collections import Counter
from copy import deepcopy
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.components.utils.constants import CLASSIFICATION, REGRESSION
from mindware.components.config_space.cs_builder import get_hpo_cs
from mindware.components.config_space.cs_builder import get_fe_cs
from mindware.components.meta_learning.fe_recomendation.train_model import calculate_relative
from openbox import History, Observation
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.config_space.space_utils import get_config_from_dict
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.utils.util_funcs import get_types

task = 'rgs'
seed = 42
rng = np.random.RandomState(seed)

task_type = CLASSIFICATION
if task == 'rgs':
    task_type = REGRESSION

if task == 'cls':
    dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_embedding.pkl', 'rb'))
    meta_data_dir = './data_cls/data/'
    surrogate_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_surrogate.pkl'
    similarity_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_similarity.pkl'
    preprocessor_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/cls_meta_dataset_preprocessor.pkl'
    algorithms = [ 'adaboost', 'extra_trees', 'gradient_boosting', 'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'lightgbm', 'logistic_regression', 'qda', 'random_forest', 'xgboost' ]
    metrics = ['acc', 'f1', 'auc']
else:
    dataset_embedding = pkl.load(open('/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_embedding.pkl', 'rb'))
    meta_data_dir = './data_rgs/data/'
    surrogate_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_surrogate.pkl'
    similarity_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_similarity.pkl'
    preprocessor_path = '/root/mindware/mindware/components/meta_learning/meta_resource/meta_dataset_vec/rgs_meta_dataset_preprocessor.pkl'
    algorithms = [ 'adaboost', 'extra_trees', 'gradient_boosting', 'k_nearest_neighbors', 'lasso_regression', 'liblinear_svr', 'libsvm_svr', 'lightgbm', 'random_forest', 'ridge_regression', 'xgboost' ]
    metrics = ['mse', 'r2', 'mae']

dataset_ids = dataset_embedding['task_ids']
dataset_ids = [tmp[5:] for tmp in dataset_ids]

surs_dict = {}
cs_dict = {}
sim_dict = {}
pre_best_dict = {}
pre_def_dict = {}

def get_cs(algo, _task_type):
    hpo_cs = get_hpo_cs(estimator_id=algo, task_type=_task_type, meta=True)
    fe_cs = get_fe_cs(_task_type, meta=True)

    tmp_cs = deepcopy(fe_cs)
    hpo_cs.add_hyperparameters(tmp_cs.get_hyperparameters())
    hpo_cs.add_conditions(tmp_cs.get_conditions())
    hpo_cs.add_forbidden_clauses(tmp_cs.get_forbiddens())

    return hpo_cs

counts = 0
num = 0
for metric in metrics:
    surs = [ [ [None] * 3 for _ in range(len(algorithms)) ] for _ in range(len(dataset_ids))]
    surs_dict[metric] = surs
    sims = np.full((len(algorithms), len(dataset_ids), len(dataset_ids)), np.nan)
    sim_dict[metric] = sims
    best = [ [ 'empty' for _ in range(len(algorithms)) ] for _ in range(len(dataset_ids))]
    pre_best_dict[metric] = best
    defa = [ [ ['empty'] for _ in range(len(algorithms)) ] for _ in range(len(dataset_ids))]
    pre_def_dict[metric] = defa

    for j, algo in enumerate(algorithms):
        print("==========", algo, "==========")
        cs = get_cs(algo, task_type)
        types, bounds = get_types(cs)
        if algo not in cs_dict:
            cs_dict[algo] = cs

        cs.seed(seed)

        # 采样25个随机样本用于预测计算相似度
        testX = []
        for i in range(100):
            testX.append(convert_configurations_to_array(
                    [cs.sample_configuration()])[0]
                )
        testX = np.array(testX)

        for i, data in enumerate(dataset_ids):
            print("-----", data, "-----")
            task_path = os.path.join(meta_data_dir, metric, algo, data)
            if not os.path.exists(task_path):
                print('Not Exists!', task_path)
                continue

            bests = []
            defas = []
            for k, file in enumerate(os.listdir(task_path)):
                sub_path = os.path.join(task_path, file)
                topk_file = None
                for tmp in os.listdir(sub_path):
                    if 'topk_config.pkl' in tmp:
                        topk_file = os.path.join(sub_path, tmp)
                        break

                if topk_file is None:
                    print("No topk!", sub_path)
                    continue

                topk = pkl.load(open(topk_file, 'rb'))
                topk = topk[list(topk.keys())[0]]

                # 训练代理模型
                his = History(config_space=cs)
                for _ob in topk:
                    config_dict = _ob[0]
                    if 'lightgbm:verbose' in config_dict:
                        config_dict.pop('lightgbm:verbose')
                    config = get_config_from_dict(config_space=cs, config_dict=config_dict)
                    perf = _ob[1]
                    obs = Observation(config, objectives=[-perf])
                    his.update_observation(obs)

                surrogate_gp = create_gp_model(
                    model_type='gp', config_space=cs, types=types, bounds=bounds, rng=rng)

                train_X = his.get_config_array()
                train_Y = his.get_objectives()
                surrogate_gp.train(train_X, train_Y)

                surs[i][j][k] = surrogate_gp


                bests.append(topk[0][0]['preprocessor'])

                # defas = []
                for t in topk:
                    p = t[0]['preprocessor']
                    if p == 'empty' and t[0]['rescaler'] == 'empty':  # and t[0]['balancer'] == 'empty' 
                        break

                    if p not in defas and p != 'empty':
                        defas.append(p)

            if len(bests) > 0:
                best[i][j] = Counter(bests).most_common(1)[0][0]
            defas.sort()
            counts += len(defas)
            num += 1
            defa[i][j] = defas


        for m in range(len(dataset_ids)):
            for n in range(len(dataset_ids)):
                s1s = surs[m][j]
                s2s = surs[n][j]

                p1s = []
                p2s = []
                for s1 in s1s:
                    if s1 is not None:
                        p1, _ = s1.predict(testX)
                        p1s.append(p1.reshape(-1))
                for s2 in s2s:
                    if s2 is not None:
                        p2, _ = s2.predict(testX)
                        p2s.append(p2.reshape(-1))

                if len(p1s) == 0 or len(p2s) == 0:
                    continue
                p1s = np.mean(p1s, axis=0)
                p2s = np.mean(p2s, axis=0)
                sims[j][m][n] = calculate_relative(p1s, p2s)

        surrogate_backup = {
            'task_ids': dataset_embedding['task_ids'],
            'algorithms_included': algorithms,
            'config_space': cs_dict,
            'surrogates': surs_dict
        }
        sim_backup = {
            'task_ids': dataset_embedding['task_ids'],
            'algorithms_included': algorithms,
            'similarity': sim_dict
        }
        with open(surrogate_path, 'wb') as f:
            pkl.dump(surrogate_backup, f)

        with open(similarity_path, 'wb') as f:
            pkl.dump(sim_backup, f)


        pre_backup = {
            'task_ids': dataset_embedding['task_ids'],
            'algorithms_included': algorithms,
            'best_preprocessor': pre_best_dict,
            'default_preprocesor': pre_def_dict,
        }
        with open(preprocessor_path, 'wb') as f:
            pkl.dump(pre_backup, f)

print(counts / num)
breakpoint()