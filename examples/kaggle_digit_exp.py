
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindware.utils.data_manager import DataManager
from mindware import CASHFE
from mindware import CASH
from mindware import CLASSIFICATION

if __name__ == '__main__':

    # Load data
    data_dir = '/Users/xubeideng/Documents/Scientific Research/AutoML/automl_data/kaggle/digit'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), label_name='label')
    train_data_node = dm.preprocess_fit(train_data_node, CLASSIFICATION)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'))
    test_data_node = dm.preprocess_transform(test_data_node)

    # Initialize CASHFE

    metric = 'acc'
    evaluation = 'holdout'
    ensemble_method = "blending"

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'liblinear_svc', 'libsvm_svc',
        'logistic_regression', 'qda', 'random_forest',
        'lightgbm'
    ]

    # 'lda',
    hpo = CASH(
        include_algorithms=include_algorithms, sub_optimizer='smac', task_type=CLASSIFICATION,
        metric=metric,
        data_node=train_data_node, evaluation=evaluation, resampling_params=None,
        optimizer='smac', inner_iter_num_per_iter=1,
        time_limit=4024, amount_of_resource=100, per_run_time_limit=300,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method=ensemble_method, ensemble_size=10
    )

    # hpo = BaseCASH(
    #     include_algorithms=include_algorithms, sub_optimizer='smac',
    #     metric=metric,
    #     data_node=train_data_node, evaluation='holdout', resampling_params=None,
    #     optimizer='mab', per_run_time_limit=600,
    #     time_limit=1024, amount_of_resource=100,
    #     output_dir='./data', seed=1, n_jobs=1,
    #     ensemble_method="blending", ensemble_size=5
    # )

    print(hpo.run())
    hpo.refit()
    pred_ens = hpo.predict(test_data_node, ens=True)
    pred = hpo.predict(test_data_node, ens=False)

    pred_ens = dm.decode_label(pred_ens)
    pred = dm.decode_label(pred)

    result = pd.DataFrame({'ImageId': list(range(1, len(pred)+1)), 'Label': pred})
    result.to_csv(os.path.join(data_dir, 'cash_smac4024_ble10_result.csv'), index=False)
    print('Result has been saved to result.csv.')
    result_ens = pd.DataFrame({'ImageId': list(range(1, len(pred)+1)), 'Label': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, 'cash_smac4024_ble10_result_ens.csv'), index=False)
    print('Ensemble result has been saved to result_ens.csv.')

    breakpoint()
