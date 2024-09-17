import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

# 关闭不必要的警告
warnings.filterwarnings('ignore')

# 加载乳腺癌数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Original data shape:", X.shape)

# 划分训练集和测试集
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 数据扩展技巧（简单版本）：重复数据
def augment_data(train_df, y_train):
    num_features = train_df.shape[1]
    augmented_train_df = pd.concat([train_df] * num_features, ignore_index=True)
    augmented_y_train = pd.concat([y_train] * num_features, ignore_index=True)
    return augmented_train_df.to_numpy(), augmented_y_train.to_numpy()

# 不使用扩展和交叉验证的情况
def train_without_cv_and_augmentation(X, y, X_test, y_test):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# 仅使用数据扩展的情况
def train_with_augmentation_no_cv(X_train, y_train, X_test, y_test):
    X_train_augmented, y_train_augmented = augment_data(X_train, y_train)
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train_augmented, y_train_augmented)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# 仅使用交叉验证的情况
def train_with_cv_no_augmentation(X_train, y_train, X_test, y_test):
    models = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    for fold, (train_indexes, valid_indexes) in enumerate(skf.split(X_train, y_train)):
        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X_train.iloc[train_indexes], y_train.iloc[train_indexes])  # 使用 iloc 进行索引
        models.append(model)

    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, binary_predictions)
    return accuracy

# 同时使用数据扩展和交叉验证的情况
def train_with_cv_and_augmentation(X_train, y_train, X_test, y_test):
    models = []
    X_train_augmented, y_train_augmented = augment_data(X_train, y_train)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    for fold, (train_indexes, valid_indexes) in enumerate(skf.split(X_train_augmented, y_train_augmented)):
        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X_train_augmented[train_indexes], y_train_augmented[train_indexes])  # 使用 iloc 进行索引
        models.append(model)

    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, binary_predictions)
    return accuracy

def main():
    results = {}

    # 使用未扩展的数据进行训练和预测
    results["Without Augmentation and CV"] = train_without_cv_and_augmentation(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # 仅使用数据扩展
    results["With Augmentation, No CV"] = train_with_augmentation_no_cv(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # 仅使用交叉验证
    results["With CV, No Augmentation"] = train_with_cv_no_augmentation(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # 使用扩展的数据和交叉验证进行训练和预测
    results["With CV and Augmentation"] = train_with_cv_and_augmentation(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    # 汇总结果
    print("\nSummary of Results:")
    for key, value in results.items():
        print(f"{key}: Test Accuracy = {value:.4f}")

if __name__ == '__main__':
    main()
