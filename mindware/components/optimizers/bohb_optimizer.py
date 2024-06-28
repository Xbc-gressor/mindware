import time
import os
import numpy as np
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.base.bohbbase import BohbBase


class BohbOptimizer(BaseOptimizer, BohbBase):
    def __init__(self, evaluator, config_space, name, eval_type, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', timestamp=None,
                 inner_iter_num_per_iter=1, seed=1,
                 R=27, eta=3, mode='smac', n_jobs=1):
        BaseOptimizer.__init__(self, evaluator, config_space, name, eval_type=eval_type, timestamp=timestamp,
                               output_dir=output_dir, seed=seed)
        BohbBase.__init__(self, eval_func=self.evaluator, config_generator=mode, config_space=self.config_space,
                          per_run_time_limit=per_run_time_limit,
                          seed=seed, R=R, eta=eta, n_jobs=n_jobs)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.inner_iter_num_per_iter = inner_iter_num_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

    def iterate(self, budget=MAX_INT):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(self.inner_iter_num_per_iter):
            _time_elapsed = time.time() - _start_time
            if _time_elapsed >= budget:
                break
            budget_left = budget - _time_elapsed
            iter_full_eval_configs, iter_full_eval_perfs = self._iterate(self.s_values[self.inner_iter_id], budget=budget_left)
            self.update_saver(iter_full_eval_configs, iter_full_eval_perfs)
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

            # Remove tmp model
            if self.evaluator.continue_training:
                for filename in os.listdir(self.evaluator.model_dir):
                    # Temporary model
                    if 'tmp_%s' % self.evaluator.timestamp in filename:
                        try:
                            filepath = os.path.join(self.evaluator.model_dir, filename)
                            os.remove(filepath)
                        except:
                            pass

        if len(self.full_eval_perfs) > 0:
            inc_idx = np.argmin(np.array(self.full_eval_perfs))
            for idx in range(len(self.full_eval_perfs)):
                if self.name in ['hpo', 'cash', 'cashfe']:
                    if hasattr(self.evaluator, 'fe_config'):
                        fe_config = self.evaluator.fe_config
                    else:
                        fe_config = None
                    self.eval_dict[(fe_config, self.full_eval_configs[idx])] = [-self.full_eval_perfs[idx], time.time()]
                else:
                    if hasattr(self.evaluator, 'hpo_config'):
                        hpo_config = self.evaluator.hpo_config
                    else:
                        hpo_config = None
                    self.eval_dict[(self.full_eval_configs[idx], hpo_config)] = [-self.full_eval_perfs[idx],
                                                                                time.time()]

            self.incumbent_perf = -self.full_eval_perfs[inc_idx]
            self.incumbent_config = self.full_eval_configs[inc_idx]

        self.perfs = [-loss for loss in self.full_eval_perfs]
        self.configs = self.full_eval_configs
        # Incumbent performance: the large, the better
        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_runtime_history(self):
        return self.full_eval_perfs, self.time_ticks, self.incumbent_perf
