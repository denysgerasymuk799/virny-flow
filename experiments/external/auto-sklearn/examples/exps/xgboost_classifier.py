from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
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


class XGBoostClassifier(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm
):
    def __init__(self,
                 max_depth,
                 learning_rate,
                 subsample,
                 min_child_weight,
                 random_state=None,
                 n_jobs=1,
                 class_weight=None):
        self.n_estimators = self.get_max_iter()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 500

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            self.max_depth = int(self.max_depth)
            self.learning_rate = float(self.learning_rate)
            self.subsample = float(self.subsample)
            self.min_child_weight = int(self.min_child_weight)

            from xgboost import XGBClassifier

            self.estimator = XGBClassifier(
                n_estimators=n_iter,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                min_child_weight=self.min_child_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
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
            "shortname": "XGB",
            "name": "XGBoost Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": False,
            # Both input and output must be tuple(iterable)
            "input": [DENSE, SIGNED_DATA, UNSIGNED_DATA],
            "output": [PREDICTIONS],
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3),
            UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True),
            UniformFloatHyperparameter("subsample", lower=0.01, upper=1.0, default_value=1.0, log=False),
            UniformIntegerHyperparameter("min_child_weight", lower=1, upper=20, default_value=1, log=False),
        ])

        return cs
