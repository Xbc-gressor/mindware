import warnings
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition
from mindware.components.utils.configspace_utils import check_for_bool, check_none
import sklearn

class FastIcaDecomposer(Transformer):
    type = 10

    def __init__(self, algorithm='parallel', whiten='False', fun='logcosh', n_components=100,
                 random_state=1):
        super().__init__("fast_ica")
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components

        self.random_state = random_state
        self.skip_flag = False
        self.pre_trained = False

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        # Skip heavy computation in fast ica.
        if X.shape[0] > 10000 or X.shape[1] > 200:
            if not self.pre_trained:
                pass
                # self.skip_flag = True
        self.pre_trained = True
        if self.skip_flag:
            return X.copy()

        if self.model is None:
            from sklearn.decomposition import FastICA

            if sklearn.__version__ <= '1.0.2':
                self.whiten = check_for_bool(self.whiten)
            else:
                if self.whiten == 'False':
                    self.whiten = False

            if check_none(self.n_components):
                self.n_components = None
            else:
                self.n_components = int(self.n_components)

            if self.n_components is not None:
                self.n_components = min(self.n_components, X.shape[0])

            self.model = FastICA(
                n_components=self.n_components, algorithm=self.algorithm,
                fun=self.fun, whiten=self.whiten, random_state=self.random_state
            )
            # Make the RuntimeWarning an Exception!
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message='array must not contain infs or NaNs')
                try:
                    self.model.fit(X)
                except ValueError as e:
                    if 'array must not contain infs or NaNs' in e.args[0]:
                        raise ValueError("Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")
                    raise e

        X_new = self.model.transform(X)
        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_components = UniformIntegerHyperparameter(
                "n_components", 10, 2000, default_value=100)
            algorithm = CategoricalHyperparameter('algorithm',
                                                  ['parallel', 'deflation'], 'parallel')
            if sklearn.__version__ <= '1.0.2':
                whiten = CategoricalHyperparameter('whiten',
                                                   ['False', 'True'], 'False')
            else:
                whiten = CategoricalHyperparameter('whiten',
                                                   ['False', 'unit-variance', 'arbitrary-variance’'], 'unit-variance')

            fun = CategoricalHyperparameter(
                'fun', ['logcosh', 'exp', 'cube'], 'logcosh')
            cs.add_hyperparameters([n_components, algorithm, whiten, fun])
            cs.add_condition(NotEqualsCondition(n_components, whiten, "False"))
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_components': hp.randint('ica_n_components', 1990) + 10,
                     'algorithm': hp.choice('ica_algorithm', ['parallel', 'deflation']),
                     'whiten': 'False',
                     'fun': hp.choice('ica_fun', ['logcosh', 'exo', 'cube'])}
            return space
