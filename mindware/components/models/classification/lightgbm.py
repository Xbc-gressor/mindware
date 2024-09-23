# 修改后
import numpy as np
import pandas as pd
from mindware.components.models.base_model import BaseClassificationModel
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, CategoricalHyperparameter, UnParametrizedHyperparameter

class LightGBM(BaseClassificationModel):
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, augment_data=0, random_state=None, verbose=0):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = int(num_leaves)
        self.max_depth = int(max_depth)
        self.subsample = subsample
        self.min_child_samples = int(min_child_samples)
        self.colsample_bytree = colsample_bytree
        self.augment_data = augment_data
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.var_mean = {}
        self.var_var = {}
        self.var_stats = {}
        self.features = None
        self.n_jobs = 4
        self.reverse_map = {}

    def fit(self, X, y):
        from lightgbm import LGBMClassifier

        print(f"Initial shape of X: {X.shape}, y: {len(y)}")

        if self.augment_data == 1:
            print("Augmenting data...")
            self.target = y
            X, y = self.augment_data_func(X, y)
            print(f"Shape after augment_data_func - X: {X.shape}, y: {len(y)}")

        self.features = X.shape[1]
        print(f"Training LightGBM model with {self.features} features")

        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose)

        if self.augment_data == 1:
            self.estimator.fit(X, y,
                               eval_set=(X, y),
                               eval_metric='auc',
                               categorical_feature=[2]
                               )
        else:
            self.estimator.fit(X, y,
                               eval_set=(X, y),
                               eval_metric='auc',
                               )
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.augment_data == 1:
            y_pred = self.predict_for_each_feature(X, mode="predict")
        else:
            y_pred = self.estimator.predict(X)
        print("predict y_pred.shape: ")
        print(y_pred.shape)
        return y_pred

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.augment_data == 1:
            y_pred = self.predict_for_each_feature(X, mode="predict_proba")
        else:
            y_pred = self.estimator.predict_proba(X)
        print("predict_proba y_pred.shape: ")
        print(y_pred.shape)
        return y_pred

    def predict_for_each_feature(self, X, mode="predict_proba"):
        if isinstance(X, pd.DataFrame):
            features = X.columns
        else:
            features = [i for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=features)

        y_preds = []

        for idx, feature in enumerate(features):
            X_single_feature = X[[feature]]

            tmp = self.var_to_feat(
                feature_data=X_single_feature[feature],
                feature_id=idx,
                is_train=False
            )

            tmp[:, 0] = (tmp[:, 0] - self.var_mean[idx]) / self.var_var[idx]

            if mode == "predict":
                y_pred = self.estimator.predict(tmp)
            else:
                y_pred = self.estimator.predict_proba(tmp)[:, 1]

            y_preds.append(y_pred)

        y_preds = np.array(y_preds)
        print(y_preds)
        y_preds = np.sum(self.logit(y_preds), axis=0)

        if mode == "predict":
            y_preds = (y_preds > 0).astype(int)
        else:
            y_preds = self.sigmoid(y_preds)

            y_preds = np.vstack([1 - y_preds, y_preds]).T

        return y_preds

    def augment_data_func(self, X, y):
        X_df = pd.DataFrame(X)
        features = X_df.columns

        print(f"Running augment_data_func, X_df shape: {X_df.shape}, y shape: {len(y)}")

        augmented_data = []
        augmented_labels = []

        for idx, feature in enumerate(features):
            print(f"Processing feature index: {idx}")

            feature_data = X_df[feature]

            if idx not in self.var_stats:
                self.var_stats[idx] = feature_data.value_counts()

            tmp = self.var_to_feat(
                feature_data=feature_data,
                feature_id=idx,
                is_train=True
            )

            self.var_mean[idx], self.var_var[idx] = np.mean(tmp[:, 0]), np.var(tmp[:, 0])

            tmp[:, 0] = (tmp[:, 0] - self.var_mean[idx]) / self.var_var[idx]

            augmented_data.append(tmp)
            augmented_labels.append(y)

        X_augmented = np.vstack(augmented_data)
        y_augmented = np.concatenate(augmented_labels)

        print(f"Final augmented data shape - X: {X_augmented.shape}, y: {len(y_augmented)}")
        return X_augmented, y_augmented

    def var_to_feat(self, feature_data, feature_id, is_train=True):
        new_df = pd.DataFrame()
        new_df["var"] = feature_data.values if isinstance(feature_data, pd.Series) else feature_data

        var_stats = self.var_stats[feature_id]

        new_df["hist"] = pd.Series(feature_data).map(var_stats)
        new_df["hist"].fillna(0, inplace=True)
        new_df["feature_id"] = feature_id
        new_df["var_rank"] = new_df["var"].rank(method='average') / len(new_df)

        if is_train and hasattr(self, 'target'):
            corr = np.corrcoef(self.target, new_df["var"])[0, 1]
            if corr < 0:
                print(f"Feature {feature_id} is negatively correlated with target. Multiplying by -1.")
                new_df["var"] = -new_df["var"]
                self.reverse_map[feature_id] = True
            else:
                self.reverse_map[feature_id] = False

        if self.reverse_map.get(feature_id, False):
            print(f"Reversing feature {feature_id} based on correlation map.")
            new_df["var"] = -new_df["var"]

        return new_df.values

    def logit(self, p):
        return np.log(p + 1e-15) - np.log(1 - p + 1e-15)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM Classifier',
                'name': 'LightGBM Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': ('SPARSE', 'DENSE', 'UNSIGNED_DATA'),
                'output': ('PREDICTIONS',)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 256, default_value=128)
        max_depth = UniformIntegerHyperparameter('max_depth', 3, 15, default_value=15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
        min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 1000, default_value=20)
        subsample = UniformFloatHyperparameter("subsample", 0.7, 1.0, default_value=1.0, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1.0, default_value=1.0, q=0.1)
        augment_data = CategoricalHyperparameter("augment_data", [0, 1], default_value=0)
        verbose = UnParametrizedHyperparameter("verbose", -1)
        cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                                colsample_bytree, augment_data, verbose])
        return cs
