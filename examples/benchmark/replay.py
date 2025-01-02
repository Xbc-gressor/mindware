import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mindware.utils.data_manager import DataManager
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.feature_engineering.parse import construct_node
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *

import json
import pickle as pkl

task_type = CLASSIFICATION

datasets_dir = '/root/automl_data/sub_automl_data/'
dataset = 'covertype'
best_info_path = '/root/mindware/examples/benchmark/newfe_ica_data/CASHFE-block_1(1)-holdout_covertype_2024-12-30-00-15-22-328567/best_model_info.json'
dataset_path = os.path.join(datasets_dir, 'cls_datasets', dataset + '.csv')


def eval(_op_list, _model, _test_node):
    
    test_data_node = _test_node.copy_()
    test_data_node = construct_node(test_data_node, _op_list)
    scorer = get_metric('acc')
    pred = _model.predict(test_data_node.data[0])
    perf = scorer._score_func(test_data_node.data[1], pred) * scorer._sign
    
    return perf
    

with open(best_info_path, 'r') as f:
    best_info = json.load(f)
    
dm = DataManager()
header = None if dataset == 'spambase' else 'infer'
label_col = 'higgs' if dataset == 'higgs' else -1

df = dm.load_csv(dataset_path, header=header)
train_df, test_df = dm.split_data(df, label_col=label_col,
                                    test_size=0.2, random_state=1, task_type=task_type)
train_data_node = dm.from_train_df(train_df, label_col=label_col)
train_data_node = dm.preprocess_fit(train_data_node, task_type)

test_data_node = dm.from_test_df(test_df, has_label=True)
test_data_node = dm.preprocess_transform(test_data_node)


ensembles = best_info['ensemble']
best = best_info['best']

op_list, best_model, _ = CombinedTopKModelSaver._load(best[2])
best_perf = eval(op_list, best_model, test_data_node)
print("best perf:", best_perf)

for i, conf in enumerate(ensembles['config']):
    w = ensembles['ensemble_weights'][i]
    op_list, best_model, p = CombinedTopKModelSaver._load(conf[2])
    perf = eval(op_list, best_model, test_data_node)
    print("ens%d:"%(i+1), "(val)%.4f"%p, "(test)%.4f"%perf, "w%f"%w)
    
breakpoint()

"""
covertype:
ens1: (val)0.8710 (test)0.8862 w0.100000
ens2: (val)0.8649 (test)0.8807 w0.080000
ens3: (val)0.8529 (test)0.8705 w0.180000
ens4: (val)0.5659 (test)0.5547 w0.380000
ens5: (val)0.4682 (test)0.4681 w0.040000
ens6: (val)0.8702 (test)0.8842 w0.020000
ens7: (val)0.8693 (test)0.8834 w0.040000
ens8: (val)0.8680 (test)0.8872 w0.100000
ens9: (val)0.5252 (test)0.5546 w0.020000
ens10: (val)0.2531 (test)0.2631 w0.040000


"""