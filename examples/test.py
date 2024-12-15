
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from mindware.components.feature_engineering.transformation_graph import DataNode


def get_increasing_sequence(data):
    """
        Return the increasing sequence.
    :param data:
    :return:
    """
    increasing_sequence = [data[0]]
    for item in data[1:]:
        _inc = increasing_sequence[-1] if item <= increasing_sequence[-1] else item
        increasing_sequence.append(_inc)
    return increasing_sequence


def is_imbalanced_dataset(labels):
    """
        Identify if this dataset is balanced or not.
    :param data_node: A DataNode object containing the dataset.
    :return: boolean. True if the dataset is imbalanced, False otherwise.
    """
    # labels = list(data_node.data[1])
    label_counts = Counter(labels)

    # Ensure there's more than one class in the dataset
    if len(label_counts) <= 1:
        raise ValueError("The dataset must contain at least two classes.")

    it = iter(label_counts.values())
    min_count = max_count = next(it)  # Initialize with the first value
    for count in it:
        if count < min_count:
            min_count = count
        elif count > max_count:
            max_count = count
    return min_count * 4 <= max_count


if '__name__' == '__main__':
    print(get_increasing_sequence([1, 2, 3, 4, 5]))