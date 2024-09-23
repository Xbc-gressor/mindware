import numpy as np#矩阵运算与科学计算的库
import pandas as pd#读取csv文件的库
#model
import lightgbm as lgb#导入lightgbm模型,这是一个集成学习模型
# #KFold是直接分成k折,StratifiedKFold还要考虑每种类别的占比
from sklearn.model_selection import StratifiedKFold
import gc#垃圾回收模块
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。

import random#提供了一些用于生成随机数的函数
#设置随机种子,保证模型可以复现
def seed_everything(seed):
    np.random.seed(seed)#numpy的随机种子
    random.seed(seed)#python内置的随机种子

seed_everything(seed=2024)
#读取训练数据和测试数据
train_df = pd.read_csv("D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\train.csv")
test_df = pd.read_csv("D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\test.csv")
train_df.head()

#除了target以外的所有特征
features = [x for x in train_df.columns if x.startswith("var")]
#对于变量特征,如果和target的相关性是负相关,则乘-1变成正相关
for var in features:
    if np.corrcoef( train_df['target'], train_df[var] )[1][0] < 0:
        train_df[var] = train_df[var] * -1
        test_df[var]  = test_df[var]  * -1

hist_df = pd.DataFrame()
for var in features:
    # 统计训练数据和测试数据total_df var列每个值出现的次数
    var_stats = pd.concat((train_df[var], test_df[var])).value_counts()
    # 测试集出现的值和出现次数的映射
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)
    # 最后判断一下var这列的值是否大于1
    hist_df[var] = (hist_df[var] > 1)

# 测试数据某行的是否有var的值在训练数据中没有出现,在测试数据只出现一次
ind = (hist_df.sum(axis=1) != 200)

#某列var 最终的value_counts统计
var_stats = {}
for var in features:
    #训练数据和测试数据中存在特有的只出现一次index进行value_counts统计.
    var_stats[var] = pd.concat((train_df[var],test_df[ind][var])).value_counts()

#变量自身,变量的count,第几个变量,归一化后变量的排名
def var_to_feat(vr, var_stats, feat_id ):
    #创建新的表格
    new_df = pd.DataFrame()
    #传入的train_df[var].values
    new_df["var"] = vr.values
    #vr对应的count
    new_df["hist"] = pd.Series(vr).map(var_stats)
    #第几个变量
    new_df["feature_id"] = feat_id
    #归一化后的变量排名
    new_df["var_rank"] = new_df["var"].rank()/len(vr)
    return new_df.values

#var_0构造特征和target,var_1构造特征和target,……搞了200倍的数据

#就是把训练数据的target*200
TARGET = np.array( list(train_df['target'].values) * 200 )
TRAIN = []
var_mean = {}
var_var  = {}
for var in features:
    #train_df[var]是数据,var_stats[var]是value_counts统计,'var_0':第几个变量
    tmp = var_to_feat(train_df[var], var_stats[var], int(var[4:]) )
    #变量本身的均值,方差
    var_mean[var],var_var[var] = np.mean(tmp[:,0]),np.var(tmp[:,0])
    #归一化后的数值
    tmp[:,0] = (tmp[:,0]-var_mean[var])/var_var[var]
    #训练数据
    TRAIN.append( tmp )
TRAIN = np.vstack( TRAIN )

#清理数据
del train_df
gc.collect()
print(f"TRAIN.shape:{TRAIN.shape},len(TARGET):{len(TARGET)}")

#模型的训练很常规,就不看了
model = lgb.LGBMClassifier(**{
    'n_estimators': 120,
    'learning_rate': 0.04,
    'num_leaves': 31,
    'min_child_samples': 1000,
    'subsample': 0.85,
    'random_state': 2024,
    'verbose': -1
})
MODELS = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
for fold, (train_indexes, valid_indexes) in enumerate(skf.split(TRAIN, TARGET)):
    print('Fold:', fold )
    model = model.fit( TRAIN[train_indexes], TARGET[train_indexes],
                      eval_set = (TRAIN[valid_indexes], TARGET[valid_indexes]),
                      eval_metric='auc',
                      categorical_feature = [2] )
    MODELS.append( model )
del TRAIN, TARGET
gc.collect()

#单调递增
def logit(p):
    return np.log(p) - np.log(1 - p)
ypred = np.zeros( (200000,200) )
for idx,var in enumerate(features):
    #构造测试数据
    tmp = var_to_feat(test_df[var], var_stats[var], int(var[4:]) )
    tmp[:,0] = (tmp[:,0]-var_mean[var])/var_var[var]
    for model_id in range(len(MODELS)):
        model = MODELS[model_id]
        ypred[:,idx] += model.predict_proba( tmp )[:,1] /len(MODELS)
ypred = np.mean( logit(ypred), axis=1 )

submission = test_df[['ID_code']]
submission['target'] = ypred
submission['target'] = submission['target'].rank() / len(test_df)
submission.to_csv('D:\\Code\\MindWare\\Data\\santander-customer-transaction-prediction\\bronze_submission.csv', index=False)
submission.head()