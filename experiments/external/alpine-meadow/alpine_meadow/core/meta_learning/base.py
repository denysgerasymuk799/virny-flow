"""Meta-Learning base classes."""

import numpy as np

from alpine_meadow.utils.performance import time_calls


class TaskTrace:
    """
    The class for storing the traces for a task, its associated meta features and history
    of pipeline runs.
    """

    def __init__(self, task_type, meta_features, run_history):
        self.task_type = task_type
        self.meta_features = meta_features
        self.run_history = run_history

        # num pipeline runs
        self._num_pipeline_runs = 0
        for run_history_json in self.run_history.values():
            self._num_pipeline_runs += len(run_history_json["data"])

    @staticmethod
    def get_meta_features_values(meta_features):
        sorted_meta_features = sorted(list(map(lambda meta_feature: (meta_feature.name, meta_feature.value),
                                               meta_features.metafeature_values.values())))
        sorted_meta_features_values = list(map(lambda x: x[1] if x[1] != '?' else None, sorted_meta_features))
        return sorted_meta_features_values

    @time_calls
    def get_feature_vector(self, other_meta_features):
        return np.array(
            [TaskTrace.get_meta_features_values(self.meta_features) +
             TaskTrace.get_meta_features_values(other_meta_features)],
            dtype=float)

    @time_calls
    def calculate_similarity(self, regressor, other_meta_features):
        """
        Calculate the distance (in terms of similarity) between two datasets based on their meta features
        """

        features = np.array(
            [TaskTrace.get_meta_features_values(self.meta_features) +
             TaskTrace.get_meta_features_values(other_meta_features)],
            dtype=float)
        return regressor.predict(features)[0]

    @property
    def num_pipeline_runs(self):
        if not hasattr(self, '_num_pipeline_runs'):  # compatibility with old versions
            self._num_pipeline_runs = 0
            for run_history_json in self.run_history.values():
                self._num_pipeline_runs += len(run_history_json["data"])
        return self._num_pipeline_runs
