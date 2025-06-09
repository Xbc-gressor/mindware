"""Selection of Pre-defined behavior spaces"""


def bs_configspace_similarity_and_loss_correlation():
    # "bs_configspace_similarity_and_loss_correlation"
    from mindware.modules.qdo_es.qdo.behavior_space import BehaviorSpace
    from mindware.modules.qdo_es.qdo.behavior_functions.basic import \
        LossCorrelationMeasure
    from mindware.modules.qdo_es.qdo.behavior_functions.implicit_diversity_metrics import ConfigSpaceGowerSimilarity

    bs = BehaviorSpace([ConfigSpaceGowerSimilarity, LossCorrelationMeasure])

    return bs
