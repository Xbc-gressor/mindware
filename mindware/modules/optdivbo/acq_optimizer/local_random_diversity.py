from mindware.modules.optdivbo.acq_optimizer.base import AcquisitionOptimizer
from mindware.modules.optdivbo.acq_optimizer.random import RandomSearch
from mindware.modules.optdivbo.acq_optimizer.local import LocalSearch
from mindware.modules.optdivbo.utils.metric import calculate_ranking
from mindware.modules.optdivbo.utils.data_loader import generate_candidate_features

import numpy as np
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb


class InterleavedLocalAndRandomSearchDiversity(AcquisitionOptimizer):
    def __init__(self, acquisition_function, config_space, rng=None, max_steps=None, n_steps_plateau_walk=10,
                 n_sls_iterations=50, rand_prob=0.3, alpha=0.1, beta=0.1):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)
        self.alpha = alpha
        self.beta = beta

    def maximize(self, observations, num_points, ens_configs, div_surrogate, random_configuration_chooser=None, **kwargs):

        next_configs_by_local_search = self.local_search._maximize(
            observations, self.n_sls_iterations, **kwargs)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            observations, num_points - len(next_configs_by_local_search),
            _sorted=True)

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of openbox. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
                next_configs_by_random_search_sorted
                + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])

        all_configs = [_[1] for _ in next_configs_by_acq_value]

        # ------------ Start ------------

        all_scores = {'acq': [], 'div': []}
        for config_tuple in next_configs_by_acq_value:
            config = config_tuple[1]
            all_scores['acq'].append(config_tuple[0])
        
        ens_size = len(ens_configs)
        config_num = len(all_configs)
        candidate_X = generate_candidate_features(ens_configs=ens_configs, candidate_configs=all_configs)
        pred_div = div_surrogate.predict(candidate_X)
        pred_div = np.array(pred_div).reshape(config_num, ens_size).min(axis = -1)
        all_scores['div'] = pred_div.tolist()

        iters = len(observations)

        all_weights = {'acq': 1.0, 'div': self.alpha * (1 / (1 + np.exp(-self.beta*iters)) - 0.5)}
        #all_weights = {'acq': 1.0, 'div': self.alpha * np.tanh(self.beta * iters)}

        # print(all_scores['acq'])
        # print(all_scores['div'])
        # print(all_weights)
        
        # Calculate the ranking of both acq function and diversity
        all_rankings = calculate_ranking(all_scores)
        
        # Combine with weights
        final_ranking_list = list()
        for i in range(num_points):
            ranking = 0
            for key in all_rankings:
                ranking += all_rankings[key][i] * all_weights[key]
            final_ranking_list.append(ranking)

        # print(final_ranking_list)

        best_indices = np.argsort(final_ranking_list)  # Ascending Order
        next_configs = [next_configs_by_acq_value[idx][1] for idx in best_indices]

        # ------------ End ------------

        challengers = ChallengerList(next_configs,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()

        return challengers


class ChallengerList(object):
    def __init__(self, challengers, configuration_space, random_configuration_chooser):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.random_configuration_chooser = random_configuration_chooser

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers):
            raise StopIteration
        else:
            if self.random_configuration_chooser.check(self._iteration):
                config = self.configuration_space.sample_configuration()
                config.origin = 'Random Search'
            else:
                config = self.challengers[self._index]
                self._index += 1
            self._iteration += 1
            return config
