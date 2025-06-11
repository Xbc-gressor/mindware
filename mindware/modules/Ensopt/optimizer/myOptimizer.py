# License: MIT

import time
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history import Observation, History
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.modules.Ensopt.advisor import MyAdvisor
from mindware.modules.ens.ens_utils import better_ens
from copy import deepcopy
from mindware.modules.ens.ens_evaluator import EnsEvaluator
from ConfigSpace import Constant
from mindware.components.utils.constants import *

class SMACEnsOptimizer(BaseOptimizer):

    def __init__(
            self, evaluator: EnsEvaluator, config_space, data_node, name, eval_type, es, ens_size: int = 5,
            time_limit=None, evaluation_limit=None,
            per_run_time_limit=300, per_run_mem_limit=1024, 
            inner_iter_num_per_iter=1, timestamp=None, 
            logging_dir='logs',
            task_id='OpenBox',
            output_dir='./', seed=1, n_jobs=1, topk=50):
        super(SMACEnsOptimizer, self).__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, 
                                            time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                            per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                            inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                            output_dir=output_dir, seed=seed, topk=topk)
        self.FAILED_PERF = [np.inf]
        self.ens_size = ens_size
        self.es = es
        self.data_node = data_node
        self.evaluator = evaluator
        self.per_run_time_limit = per_run_time_limit
        self.config_space = config_space
        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        self.config_advisor = MyAdvisor(config_space,
                                    initial_trials=4,
                                    init_strategy='default',  # default， latin_hypercube
                                    rand_prob=0.2,
                                    surrogate_type='gp',
                                    acq_type='ei',
                                    task_id=task_id,
                                    output_dir=logging_dir,
                                    random_state=self.seed,
                                    logger_kwargs=_logger_kwargs)

        self.iteration_id = 0
        self.incumbent_config = None
        self.incumbent_perf = -np.inf


    def get_history(self) -> History:
        assert self.config_advisor is not None
        return self.config_advisor.history

    def run(self) -> History:
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return np.max(self.perfs)
    
    def fetch_predict_estimator(self, task_type, estimator_id, config, X_train, y_train, weight_balance=0, data_balance=0):
        # Build the ML estimator.
        from mindware.components.utils.balancing import get_weights, smote
        _fit_params = {}
        config_dict = config.copy()
        if task_type in CLS_TASKS and weight_balance == 1:
            _init_params, fit_params = get_weights(y_train, estimator_id, None, {}, {})
            for key, val in _init_params.items():
                config_dict[key] = val
            if 'sample_weight' in fit_params:
                _fit_params['sample_weight'] = fit_params['sample_weight']
            elif data_balance == 1:
                X_train, y_train = smote(X_train, y_train)
        if task_type in CLS_TASKS:
            from mindware.components.evaluators.cls_evaluator import get_estimator
        elif task_type in RGS_TASKS:
            from mindware.components.evaluators.rgs_evaluator import get_estimator
        _, estimator = get_estimator(config_dict, estimator_id)

        estimator.fit(X_train, y_train, **_fit_params)
        return estimator
    
    def iterate(self, steps, budget=MAX_INT) -> Observation:

        _start_time = time.time()

        timeout = self.per_run_time_limit
        if np.isinf(timeout):
            timeout = None
        print(self.es.model_pool)
        # get configuration suggestion from advisor
        model_idx = steps % self.ens_size
        self.es.delete_model(model_idx)  # 删除列表内相应idx的模型
        run_histor_configs = self.config_advisor.get_history().configurations
        loss_lst = []
        for config in run_histor_configs:
            # 在evaluator内训练模型
            obj_args, obj_kwargs = (config, self.es), dict()
            result = run_obj_func(self.evaluator, obj_args, obj_kwargs, timeout)
            # 更新history中的objectives
            # parse result
            ret, timeout_status, traceback_msg, elapsed_time = (
                result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
            if timeout_status:
                trial_state = TIMEOUT
            elif traceback_msg is not None:
                trial_state = FAILED
                logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
            else:
                trial_state = SUCCESS

            if trial_state == SUCCESS:
                objectives, constraints, extra_info = parse_result(ret)
            else:
                objectives, constraints, extra_info = self.FAILED_PERF.copy(), None, None
            loss_lst.append(objectives)
        # 修改一个objective即可

        observations = []
        for obs, loss in zip(self.config_advisor.history.observations, loss_lst):
            new_obs = Observation(
            config=obs.config, objectives=loss, constraints=obs.constraints,
            trial_state=obs.trial_state, elapsed_time=obs.elapsed_time, extra_info=obs.extra_info,
        )
            observations.append(new_obs)
        self.config_advisor.history.observations = observations
        
        # 下一步采样
        config = self.config_advisor.get_suggestion()
        self.logger.info('conf: %s' % str(config.get_dictionary()))
        # evaluate configuration on objective_function
        obj_args, obj_kwargs = (config, self.es), dict()
        # TODO:把历史给到evaluator内
        result = run_obj_func(self.evaluator, obj_args, obj_kwargs, timeout)

        # parse result
        ret, timeout_status, traceback_msg, elapsed_time = (
            result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
        if timeout_status:
            trial_state = TIMEOUT
        elif traceback_msg is not None:
            trial_state = FAILED
            logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
        else:
            trial_state = SUCCESS

        if trial_state == SUCCESS:
            objectives, constraints, extra_info = parse_result(ret)
        else:
            objectives, constraints, extra_info = self.FAILED_PERF.copy(), None, None


        observation = Observation(
            config=config, objectives=objectives, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
        )
        
        self.config_advisor.update_observation(observation)

        
        loss_lst.append(objectives)
        # print(loss_lst)
        # 更新es_pool
        best_config_idx = np.argmin([loss[0] for loss in loss_lst])
        best_config = self.config_advisor.get_history().configurations[best_config_idx]
        self.es.replace_model(self.evaluator.train_estimator(best_config), model_idx)
        
        # self.update_saver([config], [objectives[0]])
        if trial_state == SUCCESS:
            self.configs.append(config)
            self.perfs.append(-objectives[0])

        run_history = self.config_advisor.get_history()
        if len(run_history.get_incumbents()) > 0:
            incumbent = run_history.get_incumbents()[0]
            self.incumbent_config, self.incumbent_perf = incumbent.config.get_dictionary().copy(), incumbent.objectives[0]
            self.incumbent_perf = -self.incumbent_perf
        iteration_cost = time.time() - _start_time

        self.iteration_id += 1
        # Logging
        if self.config_advisor.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.' % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_opt_trajectory(self):

        trajectory = {
            'detail_perfs': ",".join([str(p) for p in self.perfs]),
        }

        return trajectory

    def get_incumbent_config(self):
        incumbent_config = self.incumbent_config.copy()

        return incumbent_config
