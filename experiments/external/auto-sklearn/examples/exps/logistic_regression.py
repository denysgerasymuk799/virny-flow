from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from autosklearn.pipeline.constants import (
    DENSE,
    SIGNED_DATA,
    UNSIGNED_DATA,
    PREDICTIONS,
)


class LogisticRegression(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 penalty='l2',
                 C=1.0,
                 solver='lbfgs',
                 random_state=None):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.random_state = random_state
        self.max_iter = 1000
        self.estimator = None

    def fit(self, X, y):
        self.C = float(self.C)

        from sklearn.linear_model import LogisticRegression
        
        self.estimator = LogisticRegression(
            penalty=None if self.penalty == 'None' else self.penalty,
            C=self.C,
            solver=self.solver,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LR",
            "name": "Logistic Regression Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
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

        penalty = CategoricalHyperparameter("penalty", ['l2', 'None'])
        C = CategoricalHyperparameter("C", [0.001, 0.01, 0.1, 1])
        solver = CategoricalHyperparameter("solver", ['newton-cg', 'lbfgs', 'sag', 'saga'])

        cs.add_hyperparameters([penalty, C, solver])
        return cs
