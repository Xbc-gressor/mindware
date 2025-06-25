import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.util_funcs import get_types

cls_default_algorithm_set = ['extra_trees', 'random_forest',
                             'adaboost', 'gradient_boosting',
                             'k_nearest_neighbors', 'liblinear_svc',
                             'libsvm_svc', 'lightgbm',
                             'lda', 'qda',
                             'logistic_regression', 'random_forest']
rgs_default_algorithm_set = ['extra_trees', 'libsvm_svr', 'lightgbm', 'k_nearest_neighbors',
                             'random_forest', 'gradient_boosting', 'liblinear_svr',
                             ]


def get_space(task_type='cls', algorithm_set='all'):
    if task_type == 'cls':
        algorithms = cls_default_algorithm_set if algorithm_set == 'all' else algorithm_set
        assert isinstance(algorithms, list)
        from mindware.components.evaluators.cls_evaluator import get_fe_cs, get_cash_cs

        fe_config_space = get_fe_cs(task_type, include_preprocessors=None,
                                    if_imbal=False)
        cash_config_space = get_cash_cs(algorithms, task_type)
        cs = ConfigurationSpace()
        if fe_config_space is not None:
            cs.add_hyperparameters(fe_config_space.get_hyperparameters())
            cs.add_conditions(fe_config_space.get_conditions())
            cs.add_forbidden_clauses(fe_config_space.get_forbiddens())
        if cash_config_space is not None:
            cs.add_hyperparameters(cash_config_space.get_hyperparameters())
            cs.add_conditions(cash_config_space.get_conditions())
            cs.add_forbidden_clauses(cash_config_space.get_forbiddens())
        joint_cs = cs
    elif task_type == 'rgs':
        algorithms = rgs_default_algorithm_set if algorithm_set == 'all' else algorithm_set
        assert isinstance(algorithms, list)
        from mindware.components.evaluators.rgs_evaluator import get_fe_cs, get_cash_cs

        fe_config_space = get_fe_cs(task_type, include_preprocessors=None)
        cash_config_space = get_cash_cs(algorithms, task_type)
        cs = ConfigurationSpace()
        if fe_config_space is not None:
            cs.add_hyperparameters(fe_config_space.get_hyperparameters())
            cs.add_conditions(fe_config_space.get_conditions())
            cs.add_forbidden_clauses(fe_config_space.get_forbiddens())
        if cash_config_space is not None:
            cs.add_hyperparameters(cash_config_space.get_hyperparameters())
            cs.add_conditions(cash_config_space.get_conditions())
            cs.add_forbidden_clauses(cash_config_space.get_forbiddens())
        joint_cs = cs
    else:
        raise ValueError('Invalid task type: %s' % task_type)
    return joint_cs


def get_small_space(task_type='cls'):
    algorithms = ['random_forest', 'liblinear_svc', 'k_nearest_neighbors', 'adaboost']
    assert isinstance(algorithms, list)
    from mindware.components.evaluators.cls_evaluator import get_fe_cs, get_cash_cs

    fe_config_space = get_fe_cs(task_type='cls', include_preprocessors=['empty'],
                                if_imbal=False)
    cash_config_space = get_cash_cs(algorithms, task_type='cls')
    cs = ConfigurationSpace()
    if fe_config_space is not None:
        cs.add_hyperparameters(fe_config_space.get_hyperparameters())
        cs.add_conditions(fe_config_space.get_conditions())
        cs.add_forbidden_clauses(fe_config_space.get_forbiddens())
    if cash_config_space is not None:
        cs.add_hyperparameters(cash_config_space.get_hyperparameters())
        cs.add_conditions(cash_config_space.get_conditions())
        cs.add_forbidden_clauses(cash_config_space.get_forbiddens())
    return cs


def get_lgb_space():
    cs = ConfigurationSpace()
    n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
    # max_depth = Constant('max_depth', 15)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
    subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_samples, subsample,
                            colsample_bytree])
    return cs


def convert_configurations_to_onehot_array(configs):
    example_config = configs[0]
    space = example_config.configuration_space
    types, bounds = get_types(space)
    numerical_arrays = convert_configurations_to_array(configs)
    feature_dims = 0
    for i, type in enumerate(types):
        if type == 0:
            feature_dims += 1
        else:
            feature_dims += type
    output_arrays = np.zeros((len(configs), int(feature_dims)))
    cnt_dim = 0
    for i in range(numerical_arrays.shape[1]):
        if types[i] == 0:
            output_arrays[:, cnt_dim] = numerical_arrays[:, i]
            cnt_dim += 1
        else:
            col_dim = int(types[i])
            cat_array = np.zeros((len(configs), col_dim))
            cat_array[np.arange(cat_array.shape[0]), np.array(numerical_arrays[:, i], dtype=int)] = 1
            output_arrays[:, cnt_dim:cnt_dim + col_dim] = cat_array
            cnt_dim += col_dim
    return output_arrays
