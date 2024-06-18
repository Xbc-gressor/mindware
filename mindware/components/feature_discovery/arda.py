import pandas as pd
from sklearn.metrics._scorer import _BaseScorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from mindware.components.feature_discovery.base_feature_discovery import BaseFeaDiscovery
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS

from mindware.utils.data_manager import DataManager

import numpy as np
import math
from typing import List


def _bin_count_ranking(
        feature_importance_scores: np.ndarray, mask: np.ndarray, bin_size: int
):
    """
    Count how often the "real" features appear in front of the generated features
    :param feature_importance_scores: The rankings as determined by the ranking algorithm
    :param mask: The bit mask indicating which columns were randomly generated (True) and which ones are real features (False)
    :param bin_size: Size of the bin array, corresponds to the amount of columns in the data matrix (the amount of "real" features)
    :return:
    """

    # Get sorting indices for the rankings, flip order since we have feature importance scores
    indices = feature_importance_scores.argsort()[::-1]
    # Sort the mask, so we know where the generated columns are located in terms of ranking
    sorted_mask = mask[indices[::]]
    bins = np.zeros(bin_size)

    # Iterate through this mask until we hit a generated feature
    # Add 1 for all the original features that were in front
    for i, val in zip(indices, sorted_mask):
        if val:
            break
        else:
            bins[i] += 1

    return bins


class ARDA(BaseFeaDiscovery):
    def __init__(self, data_node,
                 data_source: List[pd.DataFrame],
                 fea_sel_method: str,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 sample_size=1000,
                 random_state=1):
        super().__init__(data_node=data_node,
                         data_source=data_source,
                         fea_sel_method=fea_sel_method,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir,
                         random_state=random_state)

        self.sample_size = sample_size

    def gen_features(self, X, eta):

        L = []
        d = X.shape[1]
        m = np.mean(X, axis=1)
        s = np.cov(X)
        self.logger.debug(f"\t\tARDA: Generate: {math.ceil(eta * d)} features")
        for i in range(math.ceil(eta * d)):
            L.append(np.random.multivariate_normal(m, s))
        result = np.array(L).T
        self.logger.debug(f"\t\tARDA: Generated {result.shape}")
        return result

    def select_features(
            self,
            X,
            y,
            tau=0.1,
            eta=0.2,
            k=10
    ):

        if self.task_type in RGS_TASKS:
            estimator = RandomForestRegressor()
        elif self.task_type in CLS_TASKS:
            estimator = RandomForestClassifier()
        else:
            raise ValueError("Task type is not supported for ARDA!")

        d = X.shape[1]
        self.logger.debug("\tARDA: Generate features")
        X = np.concatenate(
            (X, self.gen_features(X, eta)), axis=1
        )  # This gives us A' from the paper

        mask = np.zeros(X.shape[1], dtype=bool)
        mask[d:] = True  # We mark the columns that were generated
        counts = np.zeros(d)

        # Repeat process 'k' times, as in the algorithm
        self.logger.debug("\tARDA: Decide feature importance")
        for i in range(k):
            estimator.fit(X, y)
            counts += _bin_count_ranking(estimator.feature_importances_, mask, d)
        return np.arange(d)[counts / k > tau]

    def wrapper(
            self,
            data_node,
            T: List[float],
            eta=0.2,
            k=10
    ):

        if self.task_type in RGS_TASKS:
            estimator = RandomForestRegressor()
        elif self.task_type in CLS_TASKS:
            estimator = RandomForestClassifier()
        else:
            raise ValueError("Task type is not supported for ARDA!")

        last_accuracy = 0
        last_indices = []

        for t in sorted(T):
            _node = data_node.copy_()
            X, y = _node.data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2
            )
            self.logger.debug("\nARDA: Select features")
            indices = self.select_features(
                X_train, y_train, tau=t, eta=eta, k=k
            )

            # If this happens, the thresholds might have been too strict
            if len(indices) == 0:
                return last_indices

            if len(X_train.iloc[:, indices]) == 0:
                return last_indices

            self.logger.debug("ARDA: Train and score")
            estimator.fit(X_train.iloc[:, indices], y_train)
            accuracy = estimator.score(X_test.iloc[:, indices], y_test)
            if accuracy < last_accuracy:
                break
            else:
                last_accuracy = accuracy
                last_indices = indices
        return last_indices

    def discovery(self, data_node: pd.DataFrame):
        final_selected_features = []
        all_columns = []

        _node = data_node.copy_()
        left_table = _node.data[0]

        base_node_id = self.dataset_discovery.get_base_node_id()

        if self.sample_size and self.sample_size < left_table.shape[0]:
            left_table = left_table.sample(self.sample_size, random_state=self.random_state)
        budget_size = left_table.shape[0]

        left_table = left_table.add_prefix(f"{base_node_id}.")

        base_table_columns = list(left_table.columns)

        join_keys = []
        source_ids = self.dataset_discovery.discovery(data_node)
        while len(source_ids) > 0:
            feature_count = 0

            # Join every table according to the budget
            while feature_count <= budget_size and len(source_ids) > 0:
                table_id = source_ids.pop()

                # Get the keys between the base node and connected node
                join_key = self.dataset_discovery.get_relation_properties_node_name(
                    from_id=base_node_id, to_id=table_id
                )[0]
                join_prop, from_table, to_table = join_key

                if join_prop["from_label"] == to_table:
                    from_column = join_prop["to_column"]
                    to_column = join_prop["from_column"]
                else:
                    from_column = join_prop["from_column"]
                    to_column = join_prop["to_column"]

                # Read right table, aggregate on the join key (reduce to 1:1 or M:1 join) by random sampling
                right_table = self.dataset_discovery.get_table_by_id(table_id)
                right_table = right_table.groupby(to_column).sample(
                    n=1, random_state=self.random_state
                )

                # Join tables, drop the right key as we don't need it anymore
                if (
                        left_table[f"{from_table}.{from_column}"].dtype
                        != right_table[f"{to_table}.{to_column}"].dtype
                ):
                    continue

                left_on = f"{from_table}.{from_column}"
                right_on = f"{to_table}.{to_column}"
                left_table = pd.merge(
                    left_table,
                    right_table,
                    how="left",
                    left_on=left_on,
                    right_on=right_on,
                )
                join_keys.append(left_on)
                join_keys.append(right_on)

                # Update feature count (subtract 1 for the deleted right key)
                feature_count += right_table.shape[1] - 1

            # Compute the columns of the batch and create the batch dataset
            columns = set(left_table.columns) - set(all_columns) - set(base_table_columns)
            columns = list(set(columns) - set(join_keys))

            # If the algorithm failed
            if len(columns) == 0:
                continue

            # If the algorithm only selects one feature
            if len(columns) == 1:
                final_selected_features.extend(columns)
                continue

            joined_tables_batch = left_table[columns]

            # Save the computed columns
            all_columns.extend(columns)

            dm = DataManager(X=joined_tables_batch, y=_node.data[1])
            # Prepare data
            tmp_node = dm.get_data_node(X=joined_tables_batch, y=_node.data[1])
            tmp_node = dm.preprocess_fit(tmp_node)

            # Run ARDA - RIFS (Random Injection Feature Selection) algorithm
            T = list(np.arange(0.0, 1.0, 0.1))
            indices = self.wrapper(tmp_node, T=T)
            fs_X = tmp_node.iloc[:, indices].columns

            # Save the selected columns of the batch
            final_selected_features.extend(fs_X)

        return left_table, base_table_columns, final_selected_features
