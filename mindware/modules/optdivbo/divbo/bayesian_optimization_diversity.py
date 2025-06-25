from mindware.modules.optdivbo.divbo.base import Baseline
import numpy as np
import os
import time
import random
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acquisition_function.acquisition import EI
from openbox.utils.util_funcs import get_types
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.transform import gaussian_transform

from mindware.modules.optdivbo.acq_optimizer.local_random_diversity import InterleavedLocalAndRandomSearchDiversity
from mindware.modules.optdivbo.ensembles.ensemble_selection import EnsembleSelection
from mindware.modules.optdivbo.utils.data_loader import generate_pairwise_dataset

from lightgbm import LGBMRegressor


class BayesianOptimizationDiversity(Baseline):
    def __init__(self, config_space, eval_func, score_name, iter_num=200, save_dir='./results', task_name='default',
                 surrogate_type='prf', ens_size=25, scorer=None, task_type='cls', val_y_labels=None, alpha=0.05,
                 beta=0.2):
        super().__init__(config_space, eval_func, iter_num, save_dir, task_name)
        types, bounds = get_types(config_space)

        self.score_name = score_name
        
        if surrogate_type == 'gp':
            self.surrogate = create_gp_model(model_type='gp',
                                             config_space=config_space,
                                             types=types,
                                             bounds=bounds,
                                             rng=self.rng)
        elif surrogate_type == 'prf':
            self.surrogate = RandomForestWithInstances(types=types, bounds=bounds, seed=self.seed)
        else:
            raise ValueError("Surrogate type %s not supported!" % surrogate_type)

        self.acq_func = EI(self.surrogate)
        self.acq_optimizer = InterleavedLocalAndRandomSearchDiversity(acquisition_function=self.acq_func,
                                                                      config_space=config_space, rng=self.rng,
                                                                      alpha=alpha, beta=beta)

        self.init_num = 10

        self.timestamp = time.time()
        self.save_path = os.path.join(self.save_dir, '%s_%s_%d_%s.pkl' % (task_name, 'bodiv', iter_num, self.timestamp))

        # Diversity surrogate
        self.div_surrogate = LGBMRegressor(n_estimators=1000, objective='mse',
                                           learning_rate=0.1, max_depth=5, n_jobs=4)

        # Intermediate ensemble
        assert val_y_labels is not None
        self.val_y_labels = val_y_labels
        self.ens_size = ens_size
        self.scorer = scorer
        self.task_type = task_type
        self.ensemble = None
        self.e_config_list = []
        self.e_valid_list = []
        self.cmp_config_list = []

        self.random_possibility = 0.2

    def sample(self):
        num_config_evaluated = len(self.observations)

        if num_config_evaluated < self.init_num:  # Sample initial configurations randomly
            repeated_flag = True
            while repeated_flag:
                repeated_flag = False
                config = self.config_space.sample_configuration()
                for observation in self.observations:
                    if config == observation[0]:
                        repeated_flag = True
                        break
            return config

        if random.random() < self.random_possibility:
            repeated_flag = True
            while repeated_flag:
                repeated_flag = False
                config = self.config_space.sample_configuration()
                for observation in self.observations:
                    if config == observation[0]:
                        repeated_flag = True
                        break
            return config
        
        train_config1, train_config2, train_diversity = generate_pairwise_dataset(self.observations,self.score_name ,self.val_y_labels)
        train_data = np.hstack([train_config1, train_config2])

        transformed_y = gaussian_transform(train_diversity)
        self.div_surrogate.fit(train_data, transformed_y)  # TODO: Continue training

        X = convert_configurations_to_array([observation[0] for observation in self.observations])
        Y = np.array([observation[1] for observation in self.observations])

        self.surrogate.train(X, Y)

        self.acq_func.update(model=self.surrogate,
                             eta=self.incumbent_value,
                             num_data=num_config_evaluated)

        challengers = self.acq_optimizer.maximize(observations=self.observations,
                                                  num_points=5000,
                                                  ens_configs=self.cmp_config_list,
                                                  div_surrogate=self.div_surrogate)

        repeated_flag = True
        repeated_time = 0
        cur_config = None
        while repeated_flag:
            repeated_flag = False
            cur_config = challengers.challengers[repeated_time]
            for observation in self.observations:
                if cur_config == observation[0]:
                    repeated_flag = True
                    repeated_time += 1
                    break
        return cur_config

    def update(self, config, val_perf, test_perf, val_pred, test_pred, time):
        if val_perf < self.incumbent_value:
            self.incumbent_value = val_perf
            self.incumbent_config = config
        self.observations.append((config, val_perf, test_perf, val_pred, test_pred, time))

        self.e_valid_list = []
        self.e_config_list = []
        self.cmp_config_list = []
        self.ensemble = EnsembleSelection(ensemble_size=self.ens_size,
                                          task_type=self.task_type,
                                          scorer=self.scorer)

        for ob in self.observations:
            config, val_perf, test_perf, val_pred, test_pred, _ = ob
            if val_pred is not None:
                self.e_valid_list.append(val_pred)
                self.e_config_list.append(config)

        if len(self.e_valid_list) > 0:
            self.ensemble.fit(self.e_valid_list, self.val_y_labels)
            print(self.ensemble.model_idx)

            # Get configs in the intermediate ensemble
            for i in self.ensemble.model_idx:
                self.cmp_config_list.append(self.e_config_list[i])

        # # Test
        # print(self.ensemble.model_idx)
        # ens_val_pred = self.ensemble.predict(self.e_valid_list)
        # ens_val_pred = np.argmax(ens_val_pred, axis=-1)
        # print(str(self.ensemble.scorer._score_func(ens_val_pred, self.val_y_labels)))
