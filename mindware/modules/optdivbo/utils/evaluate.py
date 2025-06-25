import warnings
from mindware.components.feature_engineering.parse import parse_config, construct_node


def evaluate(config, scorer, data_node, test_node, task_type, resample_ratio, seed=1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
       
        test_size = 0.33
       
        config = config.get_dictionary().copy()
        estimator_id = config['algorithm']
        train_node = data_node.copy_()
        val_node = data_node.copy_()

        if task_type == 'cls':
            from sklearn.model_selection import StratifiedShuffleSplit
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for train_index, test_index in ss.split(data_node.data[0], data_node.data[1]):
            _x_train, _x_val = data_node.data[0][train_index], data_node.data[0][test_index]
            _y_train, _y_val = data_node.data[1][train_index], data_node.data[1][test_index]


        train_node.data = [_x_train, _y_train]
        val_node.data = [_x_val, _y_val]
        
        
        data_node, op_list = parse_config(train_node, config, record=True)
        
        
        _val_node = val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        _test_node = test_node.copy_()
        _test_node = construct_node(_test_node, op_list)

    _x_train, _y_train = data_node.data
    
    if resample_ratio != 1:
        if task_type == 'cls':
            down_ss = StratifiedShuffleSplit(n_splits=1, test_size=resample_ratio,
                                             random_state=seed)
        else:
            down_ss = ShuffleSplit(n_splits=1, test_size=resample_ratio,
                                   random_state=seed)
        for _, _val_index in down_ss.split(_x_train, _y_train):
            _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
    else:
        _act_x_train, _act_y_train = _x_train, _y_train
        _val_index = list(range(len(_x_train)))

    _x_val, _y_val = _val_node.data
    _x_test, _y_test = _test_node.data
    
    config_dict = config.copy()
    # Prepare training and initial params for classifier.
    init_params, fit_params = {}, {}
    if 'sample_weight' in fit_params:
        fit_params['sample_weight'] = fit_params['sample_weight'][_val_index]
    if data_node.data_balance == 1:
        fit_params['data_balance'] = True
   
    if task_type == 'cls':
        from mindware.components.evaluators.cls_evaluator import get_estimator
        _, estimator = get_estimator(config_dict, estimator_id)
    else:
        from mindware.components.evaluators.rgs_evaluator import get_estimator
        _, estimator = get_estimator(config_dict, estimator_id)
       
    
    from mindware.components.evaluators.evaluate_func import validation
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        _fit_params = dict()
        if fit_params:
            if 'sample_weight' in fit_params:
                _fit_params['sample_weight'] = fit_params['sample_weight']
        estimator.fit(_act_x_train, _act_y_train, **_fit_params)

        if task_type == 'cls':
            valid_pred = estimator.predict_proba(_x_val)
            test_pred = estimator.predict_proba(_x_test)
        else:
            valid_pred = estimator.predict(_x_val)
            test_pred = estimator.predict(_x_test)
        
        valid_perf = scorer(estimator, _x_val, _y_val)
        test_perf = scorer(estimator, _x_test, _y_test)
    return -valid_perf, -test_perf, valid_pred, test_pred


def lgb_evaluate(config, scorer, data_node, test_node, task_type, resample_ratio, seed=1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        test_size = 0.25
        config = config.get_dictionary().copy()
        train_node = data_node.copy_()
        val_node = data_node.copy_()

        if task_type == 'cls':
            from sklearn.model_selection import StratifiedShuffleSplit
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for train_index, test_index in ss.split(data_node.data[0], data_node.data[1]):
            _x_train, _x_val = data_node.data[0][train_index], data_node.data[0][test_index]
            _y_train, _y_val = data_node.data[1][train_index], data_node.data[1][test_index]
        train_node.data = [_x_train, _y_train]
        val_node.data = [_x_val, _y_val]

    _x_train, _y_train = train_node.data

    if resample_ratio != 1:
        if task_type == 'cls':
            down_ss = StratifiedShuffleSplit(n_splits=1, test_size=resample_ratio,
                                             random_state=seed)
        else:
            down_ss = ShuffleSplit(n_splits=1, test_size=resample_ratio,
                                   random_state=seed)
        for _, _val_index in down_ss.split(_x_train, _y_train):
            _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
    else:
        _act_x_train, _act_y_train = _x_train, _y_train
        _val_index = list(range(len(_x_train)))

    _x_val, _y_val = val_node.data
    _x_test, _y_test = test_node.data

    config_dict = config.copy()
    # Prepare training and initial params for classifier.
    init_params, fit_params = {}, {}
    if 'sample_weight' in fit_params:
        fit_params['sample_weight'] = fit_params['sample_weight'][_val_index]
    if data_node.data_balance == 1:
        fit_params['data_balance'] = True

    if task_type == 'cls':
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(**config_dict)
    else:
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(**config_dict)

    from mindware.components.evaluators.evaluate_func import validation
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        _fit_params = dict()
        if fit_params:
            if 'sample_weight' in fit_params:
                _fit_params['sample_weight'] = fit_params['sample_weight']
        estimator.fit(_act_x_train, _act_y_train, **_fit_params)

        if task_type == 'cls':
            valid_pred = estimator.predict_proba(_x_val)
            test_pred = estimator.predict_proba(_x_test)
        else:
            valid_pred = estimator.predict(_x_val)
            test_pred = estimator.predict(_x_test)
        valid_perf = scorer(estimator, _x_val, _y_val)
        test_perf = scorer(estimator, _x_test, _y_test)
    return -valid_perf, -test_perf, valid_pred, test_pred
