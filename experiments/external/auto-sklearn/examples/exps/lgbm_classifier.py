from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
from autosklearn.pipeline.constants import (
    DENSE,
    SIGNED_DATA,
    UNSIGNED_DATA,
    PREDICTIONS,
)
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
import numpy as np


class LGBMClassifier(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm
):
    def __init__(self,
                 max_depth,
                 num_leaves,
                 min_data_in_leaf,
                 random_state=None,
                 n_jobs=1,
                 class_weight=None):
        self.n_estimators = self.get_max_iter()
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 1000

    def get_current_iter(self):
        return self.estimator.n_estimators if self.estimator else 0

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from lightgbm import LGBMClassifier as LGBM

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            self.max_depth = int(self.max_depth)
            self.num_leaves = int(self.num_leaves)
            self.min_data_in_leaf = int(self.min_data_in_leaf)

            self.estimator = LGBM(
                n_estimators=n_iter,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_data_in_leaf=self.min_data_in_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                verbosity=-1,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.n_estimators
            )

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False

        return self.get_current_iter() >= self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LGBM",
            "name": "LightGBM Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": False,
            "input": [DENSE, SIGNED_DATA, UNSIGNED_DATA],
            "output": [PREDICTIONS],
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        cs.add_hyperparameters([
            CategoricalHyperparameter("max_depth", [3, 4, 5, 6, 7, 8, 9, -1]),
            CategoricalHyperparameter("num_leaves", [int(x) for x in np.linspace(20, 3000, 20)]),
            UniformIntegerHyperparameter("min_data_in_leaf", lower=100, upper=1000, q=50),
        ])

        return cs
