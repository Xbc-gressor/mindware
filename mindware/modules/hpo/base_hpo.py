import time

from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.models.base_model import BaseModel
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.optimizers import build_hpo_optimizer

from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons

from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer


from sklearn.utils.multiclass import type_of_target
from mindware.components.utils.constants import type_dict


class BaseHPOOptimizer(object):
    def __init__(self, estimator_id: str, task_type=None, scorer: str = 'acc',
                 data_node: DataNode = None, evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', per_run_time_limit=600, inner_iter_num_per_iter=1,
                 time_limit=1024, evaluation_limit=20,
                 output_dir=None, seed=None, n_jobs=1):
        self.estimator_id = estimator_id
        self.scorer = scorer
        self.data_node = data_node
        self.task_type = task_type
        self.output_dir = output_dir
        self.seed = seed

        self.timestamp = time.time()

        task_type = type_of_target(data_node.data[1])
        if task_type in type_dict:
            task_type = type_dict[task_type]
        else:
            raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type

        if self.task_type in CLS_TASKS:
            self.if_imbal = is_imbalanced_dataset(self.data_node)
        else:
            self.if_imbal = False

        _candidates = get_combined_candidtates(_classifiers, _addons)
        self.cs = _candidates[self.estimator_id].get_hyperparameter_search_space()

        # Define evaluator and optimizer
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.hpo.hpo_evaluator import HPOClassificationEvaluator
            self.evaluator = HPOClassificationEvaluator(
                estimator_id=estimator_id,
                fixed_config=None,
                scorer=self.scorer,
                data_node=data_node,
                if_imbal=self.if_imbal,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                resampling_strategy=evaluation,
                resampling_params=resampling_params)
        else:
            pass

        if evaluation == 'partial':
            optimizer_class = MfseOptimizer
        elif evaluation == 'partial_bohb':
            optimizer_class = BohbOptimizer
        else:
            # TODO: Support asynchronous BO
            if optimizer == 'random_search':
                optimizer_class = RandomSearchOptimizer
            elif optimizer == 'tpe':
                optimizer_class = TPEOptimizer
            elif optimizer == 'smac':
                optimizer_class = SMACOptimizer
            else:
                raise ValueError("Invalid optimizer %s" % optimizer)

        self.optimizer = optimizer_class(self.evaluator, self.cs, 'hpo',
                                         eval_type=evaluation, output_dir=self.output_dir,
                                         time_limit=time_limit, evaluation_limit=evaluation_limit,
                                         per_run_time_limit=per_run_time_limit,
                                         inner_iter_num_per_iter=inner_iter_num_per_iter,
                                         timestamp=self.timestamp, seed=self.seed, n_jobs=n_jobs)

    def run(self):
        incumbent_perf = self.optimizer.run()
        return incumbent_perf
