import time

import pandas as pd
import os
import sys
# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import roc_auc_score, make_scorer
from mindware.components.feature_engineering.parse import construct_node
from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
from mindware.modules.cashfe.cashfe_evaluator import CASHFECLSEvaluator
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.models.classification.lightgbm import LightGBM

from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import EnsembleBuilder
from mindware import CLASSIFICATION
import pickle as pkl
import argparse

if __name__ == '__main__':

    # 从命令行参数中解析出参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--Opt', type=str, default='cashfe', help='cash or cashfe')
    parser.add_argument('--optimizer', type=str, default='smac', help='smac or mab')
    parser.add_argument('--x_encode', type=str, default=None, help='smac or mab')
    parser.add_argument('--ensemble_method', type=str, default='blending', help='ensemble_selection or blending')
    parser.add_argument('--ensemble_size', type=int, default=10, help='ensemble size')
    parser.add_argument('--evaluation', type=str, default='holdout', help='evaluation')
    parser.add_argument('--time_limit', type=int, default=2024, help='time limit')
    parser.add_argument('--per_time_limit', type=int, default=300, help='time limit')
    args = parser.parse_args()

    Opt = args.Opt

    task_type = CLASSIFICATION

    optimizer = args.optimizer
    x_encode = args.x_encode
    ensemble_method = args.ensemble_method
    ensemble_size = args.ensemble_size
    metric = make_scorer(roc_auc_score, needs_threshold=True)
    evaluation = args.evaluation
    time_limit = args.time_limit
    per_time_limit = args.per_time_limit
    # Load data
    # data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\kaggle\\santander'
    # data_dir = 'E:\\data\\kaggle\\santander-customer-transaction-prediction'
    data_dir = '/root/automl_data/kaggle/santander'

    dm = DataManager()

    _train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), ignore_columns=['ID_code'],
                                         label_name='target')
    train_data_node = dm.preprocess_fit(_train_data_node, task_type, x_encode=x_encode)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['ID_code'])
    test_data_node = dm.preprocess_transform(test_data_node)

    # Initialize CASHFE

    include_algorithms = ['lightgbm']

    if Opt == 'cash':
        # 'lda',
        OPT = CASH
    else:
        OPT = CASHFE

    # hpo = OPT(
    #     include_algorithms=include_algorithms, sub_optimizer='smac', task_type=task_type,
    #     metric=metric,
    #     data_node=train_data_node, evaluation=evaluation, resampling_params=None,
    #     optimizer=optimizer, inner_iter_num_per_iter=1,
    #     time_limit=919200, amount_of_resource=100, per_run_time_limit=120000,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method='cross_validation', ensemble_size=ensemble_size
    # )

    evaluator = CASHFECLSEvaluator(
        fixed_config=None,
        scorer=metric,
        data_node=train_data_node,
        resampling_strategy=evaluation,
        resampling_params=None,
        timestamp=time.time(),
        output_dir='./data',
        seed=1,
        if_imbal=is_imbalanced_dataset(train_data_node))
    from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
        CategoricalHyperparameter, Constant, UnParametrizedHyperparameter


    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        cs = ConfigurationSpace()

        algorithm = CategoricalHyperparameter('algorithm', ['lightgbm'], default_value='lightgbm')
        balancer = CategoricalHyperparameter('balancer', ['empty'], default_value='empty')
        preprocessor = CategoricalHyperparameter('preprocessor', ['empty'], default_value='empty')
        rescaler = CategoricalHyperparameter('rescaler', ['empty', 'standard'], default_value='standard')

        lightgbm_augment_data = CategoricalHyperparameter("lightgbm:augment_data", [0, 1], default_value=0)
        lightgbm_colsample_bytree = UniformFloatHyperparameter("lightgbm:colsample_bytree", 0.7, 1.0, default_value=1.0)
        lightgbm_learning_rate = UniformFloatHyperparameter("lightgbm:learning_rate", 1e-3, 0.3, default_value=0.1,
                                                            log=True)
        lightgbm_max_depth = Constant('lightgbm:max_depth', 15)
        lightgbm_min_child_samples = UniformIntegerHyperparameter("lightgbm:min_child_samples", 5, 1000, default_value=20)
        lightgbm_n_estimators = UniformFloatHyperparameter("lightgbm:n_estimators", 100, 1000, default_value=500, q=50)
        lightgbm_num_leaves = UniformIntegerHyperparameter("lightgbm:num_leaves", 31, 2047, default_value=128)
        lightgbm_subsample = UniformFloatHyperparameter("lightgbm:subsample", 0.7, 1.0, default_value=1.0)
        lightgbm_verbose = UnParametrizedHyperparameter("lightgbm:verbose", -1)

        cs.add_hyperparameters([
            algorithm, balancer, preprocessor, rescaler,
            lightgbm_augment_data, lightgbm_colsample_bytree, lightgbm_learning_rate,
            lightgbm_max_depth, lightgbm_min_child_samples, lightgbm_n_estimators,
            lightgbm_num_leaves, lightgbm_subsample, lightgbm_verbose
        ])

        return cs

    cs = get_hyperparameter_search_space()

    values = {
        'algorithm': 'lightgbm',
        'balancer': 'empty',
        'lightgbm:augment_data': 1,
        'lightgbm:colsample_bytree': 1.0,
        'lightgbm:learning_rate': 0.04,
        'lightgbm:max_depth': 15,
        'lightgbm:min_child_samples': 1000,
        'lightgbm:n_estimators': 120,
        'lightgbm:num_leaves': 31,
        'lightgbm:subsample': 0.85,
        'lightgbm:verbose': -1,
        'preprocessor': 'empty',
        'rescaler': 'empty',
    }

    config = Configuration(cs, values=values)
    # evaluator(config)

    breakpoint()


    # file_path = 'E:\codes\mindware\examples\myExps\data\\2024-09-25-21-20-07-895081_bc4b94dc52a29bbd4251bbc2c9f0bce48d146705.pkl'
    file_path = '/root/mindware/examples/data/HPO(lightgbm)-smac(1)-holdout_2024-11-24-20-34-26-298565/2024-11-24-20-34-26-298565_161336064593550de978d7bee09544283d054c28.pkl'
    op_list, model, _ = pkl.load(open(file_path, 'rb'))
    breakpoint()

    # # refit
    from mindware.components.feature_engineering.parse import parse_config
    data_node, op_list = parse_config(train_data_node, config.get_dictionary(), record=True)
    # model.fit(data_node.data[0], data_node.data[1])

    _test_data_node = construct_node(test_data_node, op_list)

    y_preds = []
    X = _test_data_node.data[0][:1000]
    for idx in range(X.shape[1]):
        

        tmp = model.var_to_feat(
            feature_data=X[:, idx],
            feature_id=idx,
            is_train=False
        )

    #     y_pred = self.estimator.predict_proba(tmp)[:, 1]
    #     y_preds.append(y_pred)

    # y_preds = np.array(y_preds)
    # y_preds = np.sum(self.logit(y_preds), axis=0)

    # if mode == "predict":
    #     y_preds = (y_preds > 0).astype(int)
    # else:
    #     from scipy.stats import rankdata
    #     y_preds = self.sigmoid(y_preds)
    #     y_preds = rankdata(y_preds) / len(y_preds)
    #     y_preds = np.vstack([1 - y_preds, y_preds]).T
        y_pred = model.estimator.predict_proba(tmp)
        y_preds.append(y_pred)
        
    import numpy as np
    def logit(p):
        return np.log(p + 1e-15) - np.log(1 - p + 1e-15)

    breakpoint()


    predictions = model.predict_proba(_test_data_node.data[0])[:, 1]
    print(predictions)

    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    result_df = pd.DataFrame({
        'ID_code': test_df['ID_code'],
        'target': predictions
    })
    result_df['target'] = result_df['target'].rank() / len(test_df)
    result_df.to_csv(os.path.join(data_dir, 'predictions_with_fe_stand_refit_noauc.csv'), index=False)
    print('Predictions have been saved to predictions.csv.')

