# License: MIT

import os
import abc
import numpy as np
from datetime import datetime

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler, HaltonSampler
from openbox.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from openbox.core.base import build_acq_func, build_optimizer, build_surrogate
from openbox.core.generic_advisor import Advisor


class MyAdvisor(Advisor):
    """
    Basic Advisor Class, which adopts a policy to sample a configuration.
    """
    def alter_model(self, history: History):
        if not self.auto_alter_model:
            return

        num_config_evaluated = len(history)

        if num_config_evaluated >= 400:
            if self.surrogate_type == 'gp':
                self.surrogate_type = 'prf'
                logger.info(f'n_observations={num_config_evaluated}, change surrogate model from GP to PRF!')
                self.setup_bo_basics()

    def get_suggestion(self, history: History = None, return_list: bool = False):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history

        self.alter_model(history)

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if len(self.initial_configurations) > 0:
            res = self.initial_configurations[0]
            self.initial_configurations = self.initial_configurations[1:]
            return [res] if return_list else res
        if self.optimization_strategy == 'random':
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        if (not return_list) and self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        cY = history.get_constraints(transform='bilog')

        if self.optimization_strategy == 'bo':
            if num_config_successful < max(self.init_num, 1):
                logger.warning('No enough successful initial trials! Sample random configuration.')
                res = self.sample_random_configs(1, history)[0]
                return [res] if return_list else res

            # train surrogate model
            if self.num_objectives == 1:
                self.surrogate_model.train(X, Y[:, 0])
            elif self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objectives)
                weights = weights / np.sum(weights)
                scalarized_obj = get_chebyshev_scalarization(weights, Y)
                self.surrogate_model.train(X, scalarized_obj(Y))
            else:  # multi-objectives
                for i in range(self.num_objectives):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            for i in range(self.num_constraints):
                self.constraint_models[i].train(X, cY[:, i])

            # update acquisition function
            if self.num_objectives == 1:
                incumbent_value = history.get_incumbent_value()
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            else:  # multi-objectives
                mo_incumbent_values = history.get_mo_incumbent_values()
                if self.acq_type == 'parego':
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_values)),
                                                     num_data=num_config_evaluated)
                elif self.acq_type.startswith('ehvi'):
                    partitioning = NondominatedPartitioning(self.num_objectives, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])
                else:
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     constraint_perfs=cY,  # for MESMOC
                                                     eta=mo_incumbent_values,
                                                     num_data=num_config_evaluated,
                                                     X=X, Y=Y)

            # optimize acquisition function
            challengers = self.optimizer.maximize(runhistory=history,
                                                  num_points=5000)
            if return_list:
                # Caution: return_list doesn't contain random configs sampled according to rand_prob
                return challengers.challengers

            for config in challengers.challengers:
                if config not in history.configurations:
                    return config
            logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                'Sample random config.' % (len(challengers.challengers), ))
            return self.sample_random_configs(1, history)[0]
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)
