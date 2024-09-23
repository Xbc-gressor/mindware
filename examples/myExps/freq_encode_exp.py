import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from mindware.components.models.classification.lightgbm import LightGBM

def apply_combined_frequency_encoding(train_df, test_df, columns):
    combined_df = pd.concat([train_df, test_df])
    new_train_cols = []
    new_test_cols = []
    for col in columns:
        value_counts = combined_df[col].value_counts(dropna=False)

        train_has_one = pd.Series(0, index=train_df.index)
        test_has_one = pd.Series(0, index=test_df.index)

        train_has_one[train_df[col].map(value_counts) == 1] = 4
        test_has_one[test_df[col].map(value_counts) == 1] = 4

        train_has_one[(train_df[col].map(value_counts) > 1) & (train_df['target'] == 1)] = 1
        train_has_one[(train_df[col].map(value_counts) > 1) & (train_df['target'] == 0)] = 2

        train_has_one[train_df[col].map(value_counts) > 1] = 3
        test_has_one[test_df[col].map(value_counts) > 1] = 3

        mean_val = combined_df[col].mean()
        train_not_unique = train_df[col].copy()
        test_not_unique = test_df[col].copy()

        train_not_unique[train_df[col].map(value_counts) == 1] = mean_val
        test_not_unique[test_df[col].map(value_counts) == 1] = mean_val

        new_train_cols.append(train_has_one.rename(col + '_has_one'))
        new_train_cols.append(train_not_unique.rename(col + '_not_unique'))
        new_test_cols.append(test_has_one.rename(col + '_has_one'))
        new_test_cols.append(test_not_unique.rename(col + '_not_unique'))

    train_df = pd.concat([train_df] + new_train_cols, axis=1)
    test_df = pd.concat([test_df] + new_test_cols, axis=1)

    return train_df, test_df


def load_data():
    train_df = pd.read_csv('/Data/santander-customer-transaction-prediction/train.csv')
    test_df = pd.read_csv('/Data/santander-customer-transaction-prediction/test.csv')
    return train_df, test_df


def preprocess_data(train_df, test_df, use_frequency_encoding=False):
    y_train = train_df['target']
    train_ids = train_df['ID_code']
    test_ids = test_df['ID_code']
    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    numeric_medians = train_df[numeric_columns].median()
    train_df[numeric_columns] = train_df[numeric_columns].fillna(numeric_medians)
    test_df[numeric_columns] = test_df[numeric_columns].fillna(numeric_medians)

    columns_to_encode = numeric_columns.tolist()

    if use_frequency_encoding:
        train_df['target'] = y_train
        train_df, test_df = apply_combined_frequency_encoding(train_df, test_df, columns_to_encode)
        train_df.drop(['target'], axis=1, inplace=True)

    train_df, test_df = train_df.align(test_df, join='outer', axis=1, fill_value=0)

    return train_df, y_train, test_df, test_ids, columns_to_encode


def select_features(model, X_train, y_train):
    model.fit(X_train, y_train)
    feature_importances = model.estimator.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    return feature_importance_df


def experiment(X_train, y_train, X_test, use_frequency_encoding, columns_to_encode):
    model = LightGBM(n_estimators=200, learning_rate=0.1, num_leaves=64,
                     max_depth=15, min_child_samples=30, subsample=0.8,
                     colsample_bytree=0.8, random_state=42)

    # 选择重要特征
    feature_importance_df = select_features(model, X_train, y_train)
    important_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature']
    X_train = X_train[important_features]
    X_test = X_test[important_features]

    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return model, predictions


def save_predictions(ids, predictions, file_path):
    submission_df = pd.DataFrame({
        'ID_code': ids,
        'target': predictions
    })
    submission_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")


def main():
    train_df, test_df = load_data()

    # 不使用频率编码的预测结果
    X_train_no_fe, y_train, X_test_no_fe, test_ids, _ = preprocess_data(train_df.copy(), test_df.copy(),
                                                                        use_frequency_encoding=False)
    model, predictions_no_fe = experiment(X_train_no_fe, y_train, X_test_no_fe, False, [])
    save_predictions(test_ids, predictions_no_fe,
                     '/Data/santander-customer-transaction-prediction/santander_no_fe.csv')

    # 使用频率编码的预测结果
    X_train_with_fe, y_train, X_test_with_fe, test_ids, columns_to_encode = preprocess_data(train_df.copy(),
                                                                                            test_df.copy(),
                                                                                            use_frequency_encoding=True)
    model, predictions_with_fe = experiment(X_train_with_fe, y_train, X_test_with_fe, True, columns_to_encode)
    save_predictions(test_ids, predictions_with_fe,
                     '/Data/santander-customer-transaction-prediction/santander_with_fe.csv')


if __name__ == '__main__':
    main()
