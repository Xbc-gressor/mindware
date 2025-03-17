from sklearn.metrics._scorer import _BaseScorer
import time
import datetime
import os

from mindware.components.feature_engineering.parse import parse_config
from mindware.components.evaluators.base_evaluator import fetch_predict_estimator
from mindware.utils.logging_utils import get_logger
from mindware.components.utils.topk_saver import CombinedTopKModelSaver


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats, valid_data,
                 ensemble_method: str,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer,
                 output_dir=None, seed=None,
                 predictions=None):
        self.stats = stats
        self.valid_data = valid_data
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.output_dir = output_dir
        self.seed = seed

        self.train_labels = None
        self.datetime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)
        
        self.if_imbal = if_imbal
        self.predictions = predictions

        self.base_model_mask = None
        self.train_labels = valid_data.data[1]

    def fit(self):
        raise NotImplementedError

    def predict(self, data, refit=False):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    # TODO: Refit
    def refit(self, datanode):
        self.logger.debug("Start to refit all models needed by ensemble!")
        # Refit models on whole training data
        model_cnt = 0
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, model_path) in enumerate(model_to_eval):
                # X, y = self.node.data
                if self.base_model_mask[model_cnt] == 1:

                    save_path = CombinedTopKModelSaver.get_refit_path(model_path)
                    if os.path.exists(save_path):
                        self.logger.info("Already Refit model %d[%s], path: %s" % (model_cnt, config['algorithm'], save_path))
                        model_cnt += 1
                        continue

                    self.logger.info("Refit model %d[%s], path: %s" % (model_cnt, config['algorithm'], save_path))
                    op_list, estimator, perf = CombinedTopKModelSaver._load(model_path)

                    if op_list == {}:
                        _node = datanode.copy_()
                    else:
                        _node, op_list = parse_config(datanode.copy_(), config, record=True,
                                                      if_imbal=self.if_imbal)

                    estimator = fetch_predict_estimator(self.task_type, config['algorithm'], config,
                                                        _node.data[0], _node.data[1],
                                                        weight_balance=_node.enable_balance,
                                                        data_balance=_node.data_balance)

                    CombinedTopKModelSaver._save(items=[op_list, estimator, perf], save_path=save_path)

                model_cnt += 1
