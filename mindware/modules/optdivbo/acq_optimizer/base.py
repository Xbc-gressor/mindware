import numpy as np


class AcquisitionOptimizer:

    def __init__(self, acquisition_function, config_space, rng, task='benchmark201'):
        self.acquisition_function = acquisition_function
        self.config_space = config_space
        if rng is None:
            self.rng = np.random.RandomState(seed=42)
        else:
            self.rng = rng
        self.iter_id = 0
        self.task = task

    def maximize(self, observations, num_points, **kwargs):
        return [t[1] for t in self._maximize(observations, num_points, **kwargs)]

    def _maximize(self, observations, num_points: int, **kwargs):
        raise NotImplementedError()

    def _sort_configs_by_acq_value(self, configs):
        acq_values = self.acquisition_function(configs)
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))
        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]
