import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc
import warnings
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

TARGET = train_df['target'].values
TRAIN = train_df[features].values

# 清理数据
del train_df
gc.collect()
print(f"TRAIN.shape: {TRAIN.shape}, len(TARGET): {len(TARGET)}")

# 模型参数
params = {
    'n_estimators': 120,
    'learning_rate': 0.04,
    'num_leaves': 31,
    'min_child_samples': 1000,
    'subsample': 0.85,
    'random_state': 2024,
    'verbose': -1
}

MODELS = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
for fold, (train_indexes, valid_indexes) in enumerate(skf.split(TRAIN, TARGET)):
    print('Fold:', fold)
    model = lgb.LGBMClassifier(**params)
    model.fit(TRAIN[train_indexes], TARGET[train_indexes])
    MODELS.append(model)

del TRAIN, TARGET
gc.collect()

def logit(p):
    return np.log(p) - np.log(1 - p)

ypred = np.zeros((test_df.shape[0], len(features)))
for idx, var in enumerate(features):
    for model_id in range(len(MODELS)):
        model = MODELS[model_id]
        ypred[:, idx] += model.predict_proba(test_df[features].values)[:, 1] / len(MODELS)

ypred = np.mean(logit(ypred), axis=1)

submission = test_df[['ID_code']]
submission['target'] = ypred
submission['target'] = submission['target'].rank() / len(test_df)
submission.to_csv('D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\iron_submission.csv', index=False)
submission.head()
