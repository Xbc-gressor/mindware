import time
import sklearn
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from mindware.components.models.base_model import BaseRegressionModel, IterativeComponentWithSampleWeight
from mindware.components.utils.configspace_utils import check_none, check_for_bool
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class ExtraTreesRegressor(IterativeComponentWithSampleWeight, BaseRegressionModel):

    def __init__(self, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion

        if check_none(max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
        if check_none(max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.bootstrap = check_for_bool(bootstrap)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesRegressor as ETR

        if refit:
            self.estimator = None

        if self.estimator is None:
            max_features = int(X.shape[1] ** float(self.max_features))
            self.estimator = ETR(n_estimators=n_iter,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 bootstrap=self.bootstrap,
                                 max_features=max_features,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 oob_score=self.oob_score,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose,
                                 random_state=self.random_state,
                                 warm_start=True)

        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            if sklearn.__version__ < "1.0.2":
                criterion = CategoricalHyperparameter(
                    "criterion", ["mse", "mae"], default_value="mse")
            elif sklearn.__version__ < "1.2.2":
                criterion = CategoricalHyperparameter(
                    "criterion", ["squared_error", "absolute_error"], default_value="squared_error")
            else:
                criterion = CategoricalHyperparameter(
                    "criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"], default_value="squared_error")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

            bootstrap = CategoricalHyperparameter("bootstrap", ["True", "False"], default_value="False")
            cs.add_hyperparameters([criterion, max_features,
                                    max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    min_impurity_decrease, bootstrap])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp

            if sklearn.__version__ < "1.0.2":
                criterions = ["mse", "mae"]
            elif sklearn.__version__ < "1.2.2":
                criterions = ["squared_error", "absolute_error"]
            else:
                criterions = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

            space = {'criterion': hp.choice('et_criterion', criterions),
                     'max_features': hp.uniform('et_max_features', 0, 1),
                     'min_samples_split': hp.randint('et_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('et_min_samples_leaf,', 20) + 1,
                     'bootstrap': hp.choice('et_bootstrap', ["True", "False"])}

            init_trial = {'criterion': "mse", 'max_features': 0.5,
                          'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': "False"}
            return space
