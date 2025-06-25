from mindware.modules.optdivbo.acq_optimizer.base import AcquisitionOptimizer
from mindware.modules.optdivbo.acq_optimizer.random import RandomSearch
from mindware.modules.optdivbo.acq_optimizer.local import LocalSearch

from openbox.acq_maximizer.random_configuration_chooser import ChooserProb


class InterleavedLocalAndRandomSearch(AcquisitionOptimizer):
    def __init__(self, acquisition_function, config_space, rng=None, max_steps=None, n_steps_plateau_walk=10,
                 n_sls_iterations=50, rand_prob=0.3):
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

    def maximize(self, observations, num_points, random_configuration_chooser=None, **kwargs):
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

        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
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
