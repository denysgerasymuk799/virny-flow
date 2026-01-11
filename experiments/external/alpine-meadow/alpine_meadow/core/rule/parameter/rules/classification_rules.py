# pylint: disable=missing-docstring, invalid-name
"""Rules for specifying the hyper-parameters of classifiers."""
import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from alpine_meadow.common.proto import pipeline_pb2 as base
from ..rule import ParameterRule


class ClassificationRule(ParameterRule):
    """
    The base class for parameter rules of classifiers
    """

    def __init__(self, estimator, configuration_space):
        self.estimator = estimator
        self.configuration_space = configuration_space

    @property
    def estimator(self):
        return self.__estimator

    @estimator.setter
    def estimator(self, estimator):
        self.__estimator = estimator

    def predicate(self, task, step):
        return step.primitive == self.estimator

    def apply(self, task, step):
        step.configuration_space = self.configuration_space


class SVCRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        kernel = CategoricalHyperparameter("kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)

        configuration_space.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking, tol])

        # conditions
        # from ConfigSpace.conditions import InCondition

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        # coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

        configuration_space.add_condition(degree_depends_on_poly)

        super().__init__(base.Primitive.SVC, configuration_space)


class LinearSVCRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        penalty = CategoricalHyperparameter('penalty', ['l1', 'l2'], default_value='l2')
        loss = CategoricalHyperparameter('loss', ['hinge', 'squared_hinge'], default_value='squared_hinge')
        dual = CategoricalHyperparameter('dual', [False], default_value=False)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)

        configuration_space.add_hyperparameters([penalty, loss, dual, tol, C])

        # conditions
        # penalty_and_loss = ForbiddenAndConjunction(
        #     ForbiddenEqualsClause(penalty, "l1"),
        #     ForbiddenEqualsClause(loss, "hinge")
        # )
        # penalty_and_dual = ForbiddenAndConjunction(
        #     ForbiddenEqualsClause(dual, False),
        #     ForbiddenEqualsClause(penalty, "l1")
        # )
        # constant_penalty_and_loss = ForbiddenAndConjunction(
        #     ForbiddenEqualsClause(dual, False),
        #     ForbiddenEqualsClause(penalty, "l2"),
        #     ForbiddenEqualsClause(loss, "hinge")
        # )

        # configuration_space.add_forbidden_clauses([penalty_and_loss, penalty_and_dual, constant_penalty_and_loss])
        # configuration_space.add_forbidden_clauses([penalty_and_loss])

        super().__init__(base.Primitive.LinearSVC, configuration_space)


class LogisticRegressionRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        penalty = CategoricalHyperparameter('penalty', ['l2', 'None'])
        C = CategoricalHyperparameter("C", [0.001, 0.01, 0.1, 1])
        solver = CategoricalHyperparameter("solver", ['newton-cg', 'lbfgs', 'sag', 'saga'])
        # penalty = CategoricalHyperparameter('penalty', ['l1', 'l2'], default_value='l2')
        # dual = CategoricalHyperparameter('dual', [False], default_value=False)
        # tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        # C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        max_iter = Constant("max_iter", 1000)

        # configuration_space.add_hyperparameters([penalty, dual, tol, max_iter])
        configuration_space.add_hyperparameters([penalty, C, solver, max_iter])

        super().__init__(base.Primitive.LogisticRegression, configuration_space)


class SGDClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        loss = CategoricalHyperparameter('loss', ['log', 'modified_huber', 'hinge', 'squared_hinge', 'perceptron'],
                                         default_value='log')
        penalty = CategoricalHyperparameter('penalty', ['none', 'l1', 'l2', 'elasticnet'], default_value='l2')
        alpha = UniformFloatHyperparameter("alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
        # l1_ratio = UniformFloatHyperparameter("l1_ratio", 1e-9, 1, log=True, default_value=0.15)
        # tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True, default_value=1e-4)
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        # learning_rate = CategoricalHyperparameter("learning_rate", ["optimal", "invscaling", "constant"],
        # default_value="invscaling")
        # eta0 = UniformFloatHyperparameter("eta0", 1e-7, 1e-1, default_value=0.01)
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter("average", [False, True], default_value=False)

        configuration_space.add_hyperparameters([loss, penalty, alpha, epsilon, power_t, average])

        super().__init__(base.Primitive.SGDClassifier, configuration_space)


class RandomForestClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter('n_estimators', 50, 1000, q=50)
        max_depth = CategoricalHyperparameter("max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'None'])
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 10, q=1)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 5, q=1)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False])

        # n_estimators = UniformIntegerHyperparameter('n_estimators', 10, 100, default_value=50)
        # criterion = CategoricalHyperparameter('criterion', ['gini', 'entropy'], default_value='gini')
        # max_depth = UniformIntegerHyperparameter('max_depth', 1, 10, default_value=4)
        # min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        # min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        # bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=True)

        configuration_space.add_hyperparameters(
            [n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap])

        super().__init__(base.Primitive.RandomForestClassifier, configuration_space)


class GaussianNBRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        super().__init__(base.Primitive.GaussianNB, configuration_space)


class AdaBoostClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=500, default_value=50)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        # algorithm = CategoricalHyperparameter("algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
        # max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=1, log=False)

        configuration_space.add_hyperparameters([n_estimators, learning_rate])

        super().__init__(base.Primitive.AdaBoostClassifier, configuration_space)


class KNeighborsClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        weights = CategoricalHyperparameter("weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter("p", choices=[1, 2], default_value=2)

        configuration_space.add_hyperparameters([weights, p])

        super().__init__(base.Primitive.KNeighborsClassifier, configuration_space)

        self.updated = False

    def apply(self, task, step):
        if not self.updated:
            # hyper parameters
            meta_features = task.meta_features
            if meta_features is None:
                upper_value_for_n_neighbors = 100
            else:
                upper_value_for_n_neighbors = int(meta_features['NumberOfInstances'].value * 0.05) - 1
                upper_value_for_n_neighbors = min(100, upper_value_for_n_neighbors)
            upper_value_for_n_neighbors = max(2, upper_value_for_n_neighbors)

            n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=1, upper=upper_value_for_n_neighbors,
                                                       log=True, default_value=1)

            self.configuration_space.add_hyperparameter(n_neighbors)

            self.updated = True

        step.configuration_space = self.configuration_space


class BaggingClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=10, upper=100, default_value=10)
        max_samples = UniformFloatHyperparameter("max_samples", lower=0.5, upper=0.8, default_value=0.6)
        # max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
        # bootstrap = CategoricalHyperparameter("bootstrap", choices=[True, False], default_value=True)

        configuration_space.add_hyperparameters([n_estimators, max_samples])

        super().__init__(base.Primitive.BaggingClassifier, configuration_space)


class ExtraTreesClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=False)

        configuration_space.add_hyperparameters([
            n_estimators, criterion, max_features, min_samples_split, min_samples_leaf, bootstrap])

        super().__init__(base.Primitive.ExtraTreesClassifier, configuration_space)


class GradientBoostingClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        loss = Constant("loss", "deviance")
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3)
        criterion = CategoricalHyperparameter('criterion', ['friedman_mse', 'mse', 'mae'], default_value='mse')
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", lower=2, upper=20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", lower=1, upper=20, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", lower=0.01, upper=1.0, default_value=1.0)
        max_features = UniformFloatHyperparameter("max_features", 0.1, 1.0, default_value=1)

        configuration_space.add_hyperparameters([
            loss, learning_rate, n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf,
            subsample, max_features])

        super().__init__(base.Primitive.GradientBoostingClassifier, configuration_space)


class XGradientBoostingClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        subsample = UniformFloatHyperparameter("subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
        min_child_weight = UniformIntegerHyperparameter("min_child_weight", lower=1, upper=20, default_value=1, log=False)

        configuration_space.add_hyperparameters([max_depth, learning_rate, n_estimators, subsample, min_child_weight])

        super().__init__(base.Primitive.XGradientBoostingClassifier, configuration_space)


class LinearDiscriminantAnalysisRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)

        configuration_space.add_hyperparameter(tol)

        super().__init__(base.Primitive.LinearDiscriminantAnalysis, configuration_space)


class QuadraticDiscriminantAnalysisRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0, default_value=0.0)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)

        configuration_space.add_hyperparameters([reg_param, tol])

        super().__init__(base.Primitive.QuadraticDiscriminantAnalysis, configuration_space)


class DecisionTreeClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)

        configuration_space.add_hyperparameters([criterion, max_depth, min_samples_split, min_samples_leaf])

        super().__init__(base.Primitive.DecisionTreeClassifier, configuration_space)


class LGBMClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 1000, q=50)
        max_depth = CategoricalHyperparameter("max_depth", [3, 4, 5, 6, 7, 8, 9, -1])
        num_leaves = CategoricalHyperparameter("num_leaves", [int(x) for x in np.linspace(start = 20, stop = 3000, num = 20)])
        min_data_in_leaf = UniformIntegerHyperparameter("min_data_in_leaf", 100, 1000, q=50)

        # num_leaves = UniformIntegerHyperparameter("num_leaves", lower=2, upper=128, default_value=31)
        # max_depth = UniformIntegerHyperparameter("max_depth", lower=-1, upper=10, default_value=-1)
        # learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        # n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        n_jobs = Constant("n_jobs", 8)
        num_threads = Constant("num_threads", 8)
        verbosity = Constant("verbosity", -1)

        configuration_space.add_hyperparameters([n_estimators, max_depth, num_leaves, min_data_in_leaf, n_jobs, num_threads, verbosity])

        super().__init__(base.Primitive.LGBMClassifier, configuration_space)


class CatBoostClassifierRule(ClassificationRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        num_leaves = UniformIntegerHyperparameter("num_leaves", lower=2, upper=128, default_value=31)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=-1, upper=10, default_value=-1)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)

        configuration_space.add_hyperparameters([num_leaves, max_depth, learning_rate, n_estimators])

        super().__init__(base.Primitive.CatBoostClassifier, configuration_space)


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    rule_executor.register_parameter_rule(SVCRule())
    rule_executor.register_parameter_rule(LinearSVCRule())
    rule_executor.register_parameter_rule(LogisticRegressionRule())
    rule_executor.register_parameter_rule(SGDClassifierRule())
    rule_executor.register_parameter_rule(RandomForestClassifierRule())
    rule_executor.register_parameter_rule(GaussianNBRule())
    rule_executor.register_parameter_rule(AdaBoostClassifierRule())
    rule_executor.register_parameter_rule(KNeighborsClassifierRule())
    rule_executor.register_parameter_rule(BaggingClassifierRule())
    rule_executor.register_parameter_rule(ExtraTreesClassifierRule())
    rule_executor.register_parameter_rule(GradientBoostingClassifierRule())
    rule_executor.register_parameter_rule(XGradientBoostingClassifierRule())
    rule_executor.register_parameter_rule(LinearDiscriminantAnalysisRule())
    rule_executor.register_parameter_rule(QuadraticDiscriminantAnalysisRule())
    rule_executor.register_parameter_rule(DecisionTreeClassifierRule())
    rule_executor.register_parameter_rule(LGBMClassifierRule())
    rule_executor.register_parameter_rule(CatBoostClassifierRule())
