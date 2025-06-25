from mindware.modules.optdivbo.acq_optimizer.base import AcquisitionOptimizer


class RandomSearch(AcquisitionOptimizer):
    def _maximize(self, observations, num_points, _sorted=False, **kwargs):
        if self.task == 'nasbench101':
            from utils.config2net import bench101_config2spec
            from utils.space import bench101_opt_choices
            rand_configs = list()
            hash_list = list()
            for _ in range(num_points):
                while True:
                    config_ = self.config_space.sample_configuration()
                    spec = bench101_config2spec(config_)
                    if spec is not None:
                        model_hash = spec.hash_spec(bench101_opt_choices)
                        if model_hash not in hash_list:
                            rand_configs.append(config_)
                            hash_list.append(model_hash)
                            break

        elif self.task == 'nasbenchasr':
            from models.asr.asr_graph_utils import get_asr_model_hash
            from utils.config2net import asr_config2vec
            from utils.space import ASR_main_edge_choices
            rand_configs = list()
            hash_list = list()
            for _ in range(num_points):
                while True:
                    config_ = self.config_space.sample_configuration()
                    vec = asr_config2vec(config_)
                    vec[0][0] = ASR_main_edge_choices.index(vec[0][0])
                    vec[1][0] = ASR_main_edge_choices.index(vec[1][0])
                    vec[2][0] = ASR_main_edge_choices.index(vec[2][0])
                    model_hash = get_asr_model_hash(vec, ops=ASR_main_edge_choices)
                    if model_hash not in hash_list:
                        rand_configs.append(config_)
                        hash_list.append(model_hash)
                        break

        else:
            if num_points > 1:
                rand_configs = self.config_space.sample_configuration(
                    size=num_points)
            else:
                rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]
