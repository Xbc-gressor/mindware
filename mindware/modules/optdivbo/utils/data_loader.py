from mindware.datasets.utils import load_train_test_data
import numpy as np
from mindware.modules.optdivbo.utils.metric import diversity
from mindware.modules.optdivbo.utils.metric import opt_diversity
# from openbox.utils.config_space.util import convert_configurations_to_array
from mindware.modules.optdivbo.utils.space import convert_configurations_to_onehot_array as convert_configurations_to_array


def generate_pairwise_dataset(observations, score_name, y_true=None):
    c1 = []
    c2 = []
    div = []
    # (C1,C2,R) and (C2,C1,R) are different.
    configs = [ob[0] for ob in observations]
    config_vectors = convert_configurations_to_array(configs)
    for i, ob_i in enumerate(observations):
        for j, ob_j in enumerate(observations):
            if ob_i[3] is not None and ob_j[3] is not None:
                c1.append(config_vectors[i])
                c2.append(config_vectors[j])
                # div.append(diversity(ob_i[3], ob_j[3]))
                div.append(opt_diversity(ob_i[3], ob_j[3], score_name, y_true))
    return np.array(c1), np.array(c2), np.array(div)


def generate_leaveoneout_dataset(observations, observation):
    c1 = []
    c2 = []
    div = []
    # (C1,C2,R) and (C2,C1,R) are different.
    configs = [ob[0] for ob in observations]
    config_vectors = convert_configurations_to_array(configs)
    new_config_vector = convert_configurations_to_array([observation[0]])[0]
    for i, ob_i in enumerate(observations):
        if ob_i[3] is not None:
            c1.append(config_vectors[i])
            c2.append(new_config_vector)
            div.append(diversity(ob_i[3], observation[3]))
    return np.array(c1), np.array(c2), np.array(div)


def generate_candidate_features(ens_configs, candidate_configs):
    c1 = []
    c2 = []
    ens_config_vectors = convert_configurations_to_array(ens_configs)
    candidate_config_vectors = convert_configurations_to_array(candidate_configs)
    for v_i in candidate_config_vectors:
        for v_j in ens_config_vectors:
            c1.append(v_i)
            c2.append(v_j)
    return np.hstack([c1, c2])
