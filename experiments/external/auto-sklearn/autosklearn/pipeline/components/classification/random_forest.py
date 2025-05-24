from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
    Constant,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from autosklearn.util.common import check_for_bool, check_none


class RandomForest(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm,
):
    def __init__(
        self,
        criterion,
        max_features,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        bootstrap,
        max_leaf_nodes,
        min_impurity_decrease,
        random_state=None,
        n_jobs=1,
        class_weight=None,
    ):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 1000

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import RandomForestClassifier

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

            if self.max_features not in ("sqrt", "log2", "auto"):
                max_features = int(X.shape[1] ** float(self.max_features))
            else:
                max_features = self.max_features

            self.bootstrap = check_for_bool(self.bootstrap)

            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_impurity_decrease = float(self.min_impurity_decrease)

            # initial fit of only increment trees
            self.estimator = RandomForestClassifier(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_features=max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                warm_start=True,
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

        return not len(self.estimator.estimators_) < self.n_estimators

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
            "shortname": "RF",
            "name": "Random Forest Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        max_depth = CategoricalHyperparameter("max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'None'])
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 10, q=1)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 5, q=1)
        bootstrap = CategoricalHyperparameter("bootstrap", ["True", "False"])

        criterion = Constant(name="criterion", value="gini")
        max_features = Constant(name="max_features", value="sqrt")
        min_weight_fraction_leaf = Constant(name="min_weight_fraction_leaf", value=0.0)
        max_leaf_nodes = Constant(name="max_leaf_nodes", value="None")
        min_impurity_decrease = Constant(name="min_impurity_decrease", value=0.0)

        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                bootstrap,
                min_impurity_decrease,
            ]
        )
        return cs
