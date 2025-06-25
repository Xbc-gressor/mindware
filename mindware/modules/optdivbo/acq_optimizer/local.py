from mindware.modules.optdivbo.acq_optimizer.base import AcquisitionOptimizer

import time
from openbox.utils.config_space import get_one_exchange_neighbourhood


class LocalSearch(AcquisitionOptimizer):
    def __init__(self, acquisition_function, config_space, rng=None, task='benchmark201', max_steps=None,
                 n_steps_plateau_walk=10):
        super().__init__(acquisition_function, config_space, rng, task)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(self, observations, num_points: int, **kwargs):
        init_points = self._get_initial_points(
            num_points, observations)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            if self.task == 'nasbench101':
                from utils.config2net import bench101_config2spec
                from utils.space import bench101_opt_choices
                attempt = 0
                hash_list = list()
                while True:
                    acq_val, configuration = self._one_iter(
                        start_point, **kwargs)
                    spec = bench101_config2spec(configuration)
                    if spec is not None:
                        model_hash = spec.hash_spec(bench101_opt_choices)
                        if spec not in hash_list:
                            hash_list.append(model_hash)
                            configuration.origin = "Local Search"
                            acq_configs.append((acq_val, configuration))
                            break
                    else:
                        attempt += 1
                    if attempt == 10:
                        break

            elif self.task == 'nasbenchasr':
                from models.asr.asr_graph_utils import get_asr_model_hash
                from utils.config2net import asr_config2vec
                from utils.space import ASR_main_edge_choices
                hash_list = list()
                attempt = 0
                while True:
                    acq_val, configuration = self._one_iter(
                        start_point, **kwargs)
                    vec = asr_config2vec(configuration)
                    vec[0][0] = ASR_main_edge_choices.index(vec[0][0])
                    vec[1][0] = ASR_main_edge_choices.index(vec[1][0])
                    vec[2][0] = ASR_main_edge_choices.index(vec[2][0])
                    model_hash = get_asr_model_hash(vec, ops=ASR_main_edge_choices)
                    if model_hash not in hash_list:
                        hash_list.append(model_hash)
                        configuration.origin = "Local Search"
                        acq_configs.append((acq_val, configuration))
                        break
                    else:
                        attempt += 1
                    if attempt == 10:
                        break

            else:
                acq_val, configuration = self._one_iter(
                    start_point, **kwargs)

                configuration.origin = "Local Search"
                acq_configs.append((acq_val, configuration))

        if len(acq_configs) == 0:
            while True:
                config_ = self.config_space.sample_configuration()
                if bench101_config2spec(config_) is not None:
                    break
            acq_val = self.acquisition_function([config_])[0]
            acq_configs.append((acq_val, config_))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        return acq_configs

    def _get_initial_points(self, num_points, observations):

        if len(observations) == 0:
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = [observation[0] for observation in observations]
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(self, start_point, **kwargs):
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], **kwargs)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:
            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                print(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.seed())

            for neighbor in all_neighbors:
                s_time = time.time()
                acq_val = self.acquisition_function([neighbor], **kwargs)
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent:
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and
                     local_search_steps == self.max_steps):
                break

        return acq_val_incumbent, incumbent
