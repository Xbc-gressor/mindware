
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from mindware.utils.data_manager import DataManager
from mindware.modules.cashfe.base_cashfe import BaseCASHFE
from mindware.modules.cash.base_cash import BaseCASH

if __name__ == '__main__':

    # Load data
    data_dir = 'D:\\xbc\\Fighting\\AutoML\\datas\\titanic\\'

    dm = DataManager()

    train_data_node = dm.load_train_csv(os.path.join(data_dir, 'train.csv'), label_name='Survived', ignore_columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    train_data_node = dm.preprocess_fit(train_data_node)

    test_data_node = dm.load_test_csv(os.path.join(data_dir, 'test.csv'), ignore_columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_data_node = dm.preprocess_transform(test_data_node)

    # Initialize CASHFE

    metric = 'acc'
    resampling_strategy = 'holdout'

    include_algorithms = [
        'adaboost', 'extra_trees', 'gradient_boosting',
        'k_nearest_neighbors', 'liblinear_svc', 'libsvm_svc',
        'logistic_regression', 'qda', 'random_forest',
        'lightgbm'
    ]

    # 'lda',
    hpo = BaseCASHFE(
        include_algorithms=include_algorithms, sub_optimizer='smac',
        metric=metric,
        data_node=train_data_node, evaluation='cv', resampling_params=None,
        optimizer='smac', per_run_time_limit=600,
        time_limit=1024, amount_of_resource=100,
        output_dir='./data', seed=1, n_jobs=1,
        ensemble_method="blending", ensemble_size=5
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
    passenger_id = pd.read_csv(os.path.join(data_dir, 'test.csv'))['PassengerId']
    result = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred})
    result.to_csv(os.path.join(data_dir, 'result.csv'), index=False)
    print('Result has been saved to result.csv.')
    result_ens = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_ens})
    result_ens.to_csv(os.path.join(data_dir, 'result_ens.csv'), index=False)
    print('Ensemble result has been saved to result_ens.csv.')
