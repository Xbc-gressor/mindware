import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import gc
import warnings
from mindware.components.models.classification.lightgbm import LightGBM
import random

warnings.filterwarnings('ignore')

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed=2024)

train_df = pd.read_csv("/Data/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("/Data/santander-customer-transaction-prediction/test.csv")

features = [x for x in train_df.columns if x.startswith("var")]
for var in features:
    if np.corrcoef(train_df['target'], train_df[var])[1][0] < 0:
        train_df[var] = train_df[var] * -1
        test_df[var]  = test_df[var]  * -1

hist_df = pd.DataFrame()
for var in features:
    var_stats = pd.concat((train_df[var], test_df[var])).value_counts()
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)
    hist_df[var] = (hist_df[var] > 1)

ind = (hist_df.sum(axis=1) != 200)

var_stats = {}
for var in features:
    var_stats[var] = pd.concat((train_df[var], test_df[ind][var])).value_counts()

def var_to_feat(vr, var_stats, feat_id):
    new_df = pd.DataFrame()
    new_df["var"] = vr.values
    new_df["hist"] = pd.Series(vr).map(var_stats)
    new_df["feature_id"] = feat_id
    new_df["var_rank"] = new_df["var"].rank() / len(vr)
    return new_df.values

TARGET = np.array(list(train_df['target'].values) * 200)
TRAIN = []
var_mean = {}
var_var  = {}
for var in features:
    tmp = var_to_feat(train_df[var], var_stats[var], int(var[4:]))
    var_mean[var], var_var[var] = np.mean(tmp[:, 0]), np.var(tmp[:, 0])
    tmp[:, 0] = (tmp[:, 0] - var_mean[var]) / var_var[var]
    TRAIN.append(tmp)
TRAIN = np.vstack(TRAIN)

del train_df
gc.collect()
print(f"TRAIN.shape: {TRAIN.shape}, len(TARGET): {len(TARGET)}")

# 初始参数
params = {
    'n_estimators': 120,
    'learning_rate': 0.04,
    'num_leaves': 31,
    'max_depth': 15,
    'min_child_samples': 1000,
    'subsample': 0.85,
    'colsample_bytree': 1.0,
    'random_state': 2024,
    'verbose': -1,
}

# 扩充参数
reference_params = {
    'max_bin': 1023,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
    'feature_fraction': 1.0,
    'bagging_freq': 1,
    'bagging_fraction': 0.85,
    'objective': 'binary',
    'n_jobs': -1
}

def run_experiment(params):
    MODELS = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    for fold, (train_indexes, valid_indexes) in enumerate(skf.split(TRAIN, TARGET)):
        print('Fold:', fold)
        model = lgb.LGBMClassifier(**params)
        model.fit(TRAIN[train_indexes], TARGET[train_indexes])
        MODELS.append(model)

    return MODELS

# 进行初始实验
initial_models = run_experiment(params)

# 添加参考参数进行实验
for key, value in reference_params.items():
    params[key] = value
    print(f"Testing with parameter {key} = {value}")
    models = run_experiment(params)

# 单调递增函数
def logit(p):
    return np.log(p) - np.log(1 - p)

# 进行预测
ypred = np.zeros((200000, 200))
for idx, var in enumerate(features):
    tmp = var_to_feat(test_df[var], var_stats[var], int(var[4:]))
    tmp[:, 0] = (tmp[:, 0] - var_mean[var]) / var_var[var]
    for model_id in range(len(initial_models)):
        model = initial_models[model_id]
        ypred[:, idx] += model.predict_proba(tmp)[:, 1] / len(initial_models)
ypred = np.mean(logit(ypred), axis=1)

submission = test_df[['ID_code']]
submission['target'] = ypred
submission['target'] = submission['target'].rank() / len(test_df)
submission.to_csv('D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\initial_submission.csv', index=False)
submission.head()

# 对于每个添加的参数进行预测
for key in reference_params.keys():
    ypred = np.zeros((200000, 200))
    for idx, var in enumerate(features):
        tmp = var_to_feat(test_df[var], var_stats[var], int(var[4:]))
        tmp[:, 0] = (tmp[:, 0] - var_mean[var]) / var_var[var]
        for model_id in range(len(models)):
            model = models[model_id]
            ypred[:, idx] += model.predict_proba(tmp)[:, 1] / len(models)
    ypred = np.mean(logit(ypred), axis=1)

    submission = test_df[['ID_code']]
    submission['target'] = ypred
    submission['target'] = submission['target'].rank() / len(test_df)
    submission.to_csv(f'D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\submission_with_{key}.csv', index=False)
    submission.head()
