import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm
import numpy as np
import gc

def apply_combined_frequency_encoding(train_df, test_df, columns):
    combined_df = pd.concat([train_df, test_df])
    for col in columns:
        freq = combined_df[col].value_counts(dropna=False) / len(combined_df)
        train_df[col + '_freq'] = train_df[col].map(freq)
        test_df[col + '_freq'] = test_df[col].map(freq)
        print(f'added frequency encoding for {col}')
    return train_df, test_df

def load_data():
    train_df = pd.read_csv('/Data/house-prices-advanced-regression-techniques/train.csv')
    test_df = pd.read_csv('/Data/house-prices-advanced-regression-techniques/test.csv')
    return train_df, test_df

def preprocess_data(train_df, test_df, use_frequency_encoding=False):
    y_train = train_df['SalePrice']
    train_ids = train_df['Id']
    test_ids = test_df['Id']
    train_df.drop(['Id', 'SalePrice'], axis=1, inplace=True)
    test_df.drop(['Id'], axis=1, inplace=True)

    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    numeric_medians = train_df[numeric_columns].median()
    train_df[numeric_columns] = train_df[numeric_columns].fillna(numeric_medians)
    test_df[numeric_columns] = test_df[numeric_columns].fillna(numeric_medians)

    # 保存原始的分类列
    original_categorical_columns = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 打印每个分类特征的不同类别数
    for col in original_categorical_columns:
        print(f'Feature: {col}, Unique categories: {pd.concat([train_df[col], test_df[col]]).nunique()}')

    if use_frequency_encoding:
        train_df, test_df = apply_combined_frequency_encoding(train_df, test_df, original_categorical_columns)

    categorical_columns = train_df.select_dtypes(include=['object', 'category']).columns
    train_df = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)
    train_df, test_df = train_df.align(test_df, join='outer', axis=1, fill_value=0)

    return train_df, y_train, test_df, test_ids, original_categorical_columns

def augment_data(train_df, y_train):
    num_features = train_df.shape[1]
    augmented_train_df = pd.concat([train_df] * num_features, ignore_index=True)
    augmented_y_train = pd.concat([y_train] * num_features, ignore_index=True)
    return augmented_train_df.to_numpy(), augmented_y_train.to_numpy()

def experiment(X_train, y_train, X_test, use_frequency_encoding, categorical_columns):
    models = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2024)
    for fold, (train_indexes, valid_indexes) in enumerate(skf.split(X_train, y_train)):
        print('Fold:', fold)
        model = lightgbm.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                         min_child_weight=1, subsample=1, colsample_bytree=1,
                         reg_alpha=0.1, reg_lambda=0.1, random_state=42)
        model.fit(X_train[train_indexes], y_train[train_indexes])
        models.append(model)
    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    return models, predictions

def save_predictions(ids, predictions, file_path):
    submission_df = pd.DataFrame({
        'Id': ids,
        'SalePrice': predictions
    })
    submission_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def main():
    train_df, test_df = load_data()

    # 不使用频率编码的预测结果
    X_train_no_fe, y_train, X_test_no_fe, test_ids, _ = preprocess_data(train_df.copy(), test_df.copy(), use_frequency_encoding=False)
    X_train_no_fe_augmented, y_train_augmented = augment_data(X_train_no_fe, y_train)
    models_no_fe, predictions_no_fe = experiment(X_train_no_fe_augmented, y_train_augmented, X_test_no_fe.to_numpy(), False, [])
    save_predictions(test_ids, predictions_no_fe,
                     '/Data/house-prices-advanced-regression-techniques/house_price_predictions_no_fe.csv')

    # 使用频率编码的预测结果
    X_train_with_fe, y_train, X_test_with_fe, test_ids, categorical_columns = preprocess_data(train_df.copy(), test_df.copy(), use_frequency_encoding=True)
    X_train_with_fe_augmented, y_train_augmented = augment_data(X_train_with_fe, y_train)
    models_with_fe, predictions_with_fe = experiment(X_train_with_fe_augmented, y_train_augmented, X_test_with_fe.to_numpy(), True, categorical_columns)
    save_predictions(test_ids, predictions_with_fe,
                     '/Data/house-prices-advanced-regression-techniques/house_price_predictions_with_fe.csv')

if __name__ == '__main__':
    main()
