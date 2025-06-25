from ConfigSpace import ConfigurationSpace
import numpy as np
import pickle as pkl
import time

from mindware.modules.optdivbo.utils.time import time_limit, TimeoutException


class Baseline:
    def __init__(self, config_space: ConfigurationSpace, eval_func, iter_num=200, save_dir='./results',
                 task_name='default'):
        self.config_space = config_space
        self.eval_func = eval_func
        self.iter_num = iter_num
        self.observations = list()
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        self.incumbent_value = np.inf
        self.incumbent_config = None
        self.save_dir = save_dir
        self.task_name = task_name
        self.save_path = None

    def sample(self):
        raise NotImplementedError

    def update(self, config, val_perf, test_perf, val_pred, test_pred, time):
        if val_perf < self.incumbent_value:
            self.incumbent_value = val_perf
            self.incumbent_config = config
        self.observations.append((config, val_perf, test_perf, val_pred, test_pred, time))

    def run(self, time_limit_per_trial=30, total_time_limit=None):

        start = time.time()
        for iter in range(self.iter_num):
            config = self.sample()
            start_time = time.time()
            try:
                with time_limit(time_limit_per_trial):
                    val_obj, test_obj, val_pred, test_pred = self.eval_func(config)
                runtime = time.time() - start_time
                print('Iter: %d, Obj: %f, Test obj: %f, Eval time: %f' % (iter, val_obj, test_obj, runtime))
            except TimeoutException as e:
                print('Time out!')
                val_obj, test_obj, val_pred, test_pred = np.inf, np.inf, None, None
                runtime = time.time() - start_time
                print('Iter: %d, Failed Obj: %f, Test obj: %f, Eval time: %f' % (iter, val_obj, test_obj, runtime))
            except Exception as e:
                print(e)
                val_obj, test_obj, val_pred, test_pred = np.inf, np.inf, None, None
                runtime = time.time() - start_time
                print('Iter: %d, Failed Obj: %f, Test obj: %f, Eval time: %f' % (iter, val_obj, test_obj, runtime))
            self.update(config, val_obj, test_obj, val_pred, test_pred, runtime)
            with open(self.save_path, 'wb') as f:
                pkl.dump(self.observations, f)

            if total_time_limit is not None and time.time() - start >= total_time_limit:
                break

    @staticmethod
    def return_sequence(observations, end_time, split, nan=1):
        x = list(range(0, end_time, split))
        y = list()
        cur_ob_idx = 0
        ob_time = observations[cur_ob_idx][-1]
        for cur_time in x:
            if cur_ob_idx == len(observations):
                y.append(observations[cur_ob_idx - 1][2])
                continue

            if cur_time > ob_time:
                ob_time += observations[cur_ob_idx][-1]
                cur_ob_idx += 1

            if cur_ob_idx == 0:
                y.append(nan)
            else:
                y.append(observations[cur_ob_idx - 1][2])

        y = [min(y[:(i + 1)]) for i in range(len(y))]
        return x, y

    @staticmethod
    def return_trial_sequence(observations, end_trial=200):
        x = list(range(len(observations[:end_trial])))
        y = list()
        for observation in observations[:end_trial]:
            y.append(observation[2])

        y = [min(y[:(i + 1)]) for i in range(len(y))]
        return x, y
