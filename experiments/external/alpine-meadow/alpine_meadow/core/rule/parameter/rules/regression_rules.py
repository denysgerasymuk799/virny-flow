# pylint: disable=missing-docstring, invalid-name
"""Rules for specifying the hyper-parameters of regressors."""
import copy

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from alpine_meadow.common.proto import pipeline_pb2 as base
from ..rule import ParameterRule


class RegressionRule(ParameterRule):
    """
    The base class for parameter rules of regressors
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
        step.configuration_space = copy.deepcopy(self.configuration_space)


class SVRRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        C = UniformFloatHyperparameter("C", lower=0.03125, upper=32768, log=True, default_value=1.0)
        epsilon = UniformFloatHyperparameter("epsilon", lower=0.001, upper=1, default_value=0.1, log=True)
        kernel = CategoricalHyperparameter("kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'], default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", lower=2, upper=5, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", lower=-1, upper=1, default_value=0)
        shrinking = CategoricalHyperparameter("shrinking", choices=[True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)

        configuration_space.add_hyperparameters([C, epsilon, kernel, degree, gamma, coef0, shrinking, tol])

        # conditions
        degree_depends_on_kernel = InCondition(child=degree, parent=kernel, values=('poly', 'rbf', 'sigmoid'))
        # gamma_depends_on_kernel = InCondition(child=gamma, parent=kernel, values=('poly', 'rbf'))
        # coef0_depends_on_kernel = InCondition(child=coef0, parent=kernel, values=('poly', 'sigmoid'))

        configuration_space.add_condition(degree_depends_on_kernel)

        super().__init__(base.Primitive.SVR, configuration_space)


class LinearSVRRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        loss = CategoricalHyperparameter("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"],
                                         default_value="squared_epsilon_insensitive")
        epsilon = UniformFloatHyperparameter("epsilon", lower=0.001, upper=1, default_value=0.1, log=True)
        dual = CategoricalHyperparameter('dual', [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)

        configuration_space.add_hyperparameters([C, loss, epsilon, dual, tol])

        # conditions
        dual_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, False),
            ForbiddenEqualsClause(loss, "epsilon_insensitive")
        )

        configuration_space.add_forbidden_clause(dual_and_loss)

        super().__init__(base.Primitive.LinearSVR, configuration_space)


class LinearRegressionRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        fit_intercept = CategoricalHyperparameter('fit_intercept', [True, False], default_value=True)
        normalize = CategoricalHyperparameter('normalize', [True, False], default_value=True)

        configuration_space.add_hyperparameters([fit_intercept, normalize])

        super().__init__(base.Primitive.LinearRegression, configuration_space)


class RidgeRegressionRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # alpha = UniformFloatHyperparameter("alpha", 1e-5, 10., log=True, default_value=1.)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)

        configuration_space.add_hyperparameter(tol)

        super().__init__(base.Primitive.Ridge, configuration_space)


class SGDRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        loss = CategoricalHyperparameter(
            "loss", ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            default_value="squared_loss")
        penalty = CategoricalHyperparameter('penalty', ['none', 'l1', 'l2', 'elasticnet'], default_value='l2')
        alpha = UniformFloatHyperparameter("alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
        # l1_ratio = UniformFloatHyperparameter("l1_ratio", 1e-9, 1., log=True, default_value=0.15)
        # tol = UniformFloatHyperparameter("tol", 1e-4, 1e-1, default_value=1e-3, log=True)
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=0.1, log=True)
        learning_rate = CategoricalHyperparameter("learning_rate", ["optimal", "invscaling", "constant"],
                                                  default_value="invscaling")
        # eta0 = UniformFloatHyperparameter("eta0", 1e-7, 1e-1, default_value=0.01)
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter("average", [False, True], default_value=False)

        configuration_space.add_hyperparameters([loss, penalty, alpha, epsilon, learning_rate, power_t, average])

        super().__init__(base.Primitive.SGDRegressor, configuration_space)


class RandomForestRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter('n_estimators', 10, 100, default_value=50)
        criterion = CategoricalHyperparameter('criterion', ['mse', 'friedman_mse', 'mae'], default_value='mse')
        max_depth = UniformIntegerHyperparameter('max_depth', 1, 10, default_value=4)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=True)

        configuration_space.add_hyperparameters([n_estimators, criterion, max_depth,
                                                 min_samples_split, min_samples_leaf, bootstrap])

        super().__init__(base.Primitive.RandomForestRegressor, configuration_space)


class GaussianProcessRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # alpha = UniformFloatHyperparameter("alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)
        alpha = UniformFloatHyperparameter("alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=False)
        # thetaL = UniformFloatHyperparameter("thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True)
        # thetaU = UniformFloatHyperparameter("thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True)

        configuration_space.add_hyperparameter(alpha)

        super().__init__(base.Primitive.GaussianProcessRegressor, configuration_space)


class AdaBoostRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=500, default_value=50)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        # loss = CategoricalHyperparameter("loss", choices=["linear", "square", "exponential"], default_value="linear")
        # max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=1, log=False)

        configuration_space.add_hyperparameters([n_estimators, learning_rate])

        super().__init__(base.Primitive.AdaBoostRegressor, configuration_space)


class KNeighborsRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        weights = CategoricalHyperparameter("weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter("p", choices=[1, 2], default_value=2)

        configuration_space.add_hyperparameters([weights, p])

        super().__init__(base.Primitive.KNeighborsRegressor, configuration_space)

    def apply(self, task, step):
        configuration_space = copy.deepcopy(self.configuration_space)

        # hyper parameters
        meta_features = task.meta_features
        if meta_features is None:
            upper_value_for_n_neighbors = 100
        else:
            upper_value_for_n_neighbors = int(meta_features['NumberOfInstances'].value * 0.05) - 1
            upper_value_for_n_neighbors = min(100, max(upper_value_for_n_neighbors, 2))

        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=1, upper=upper_value_for_n_neighbors,
                                                   log=True, default_value=1)

        configuration_space.add_hyperparameter(n_neighbors)

        step.configuration_space = configuration_space


class BaggingRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = UniformIntegerHyperparameter("n_estimators", lower=10, upper=100, default_value=10)
        max_samples = UniformFloatHyperparameter("max_samples", lower=0.5, upper=0.8, default_value=0.6)
        # max_features = UniformFloatHyperparameter("max_features", 0.5, 1.0, default_value=1.0)
        # bootstrap = CategoricalHyperparameter("bootstrap", choices=[True, False], default_value=True)

        configuration_space.add_hyperparameters([n_estimators, max_samples])

        super().__init__(base.Primitive.BaggingRegressor, configuration_space)


class ExtraTreesRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter("criterion", ['mse', 'friedman_mse', 'mae'])
        max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=False)

        configuration_space.add_hyperparameters([n_estimators, criterion, max_features,
                                                 min_samples_split, min_samples_leaf, bootstrap])

        super().__init__(base.Primitive.ExtraTreesRegressor, configuration_space)


class GradientBoostingRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        loss = CategoricalHyperparameter("loss", ["ls", "lad", "huber", "quantile"], default_value="ls")
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", lower=2, upper=20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", lower=1, upper=20, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", lower=0.01, upper=1.0, default_value=1.0)
        max_features = UniformFloatHyperparameter("max_features", 0.1, 1.0, default_value=1)
        alpha = UniformFloatHyperparameter("alpha", lower=0.75, upper=0.99, default_value=0.9)

        configuration_space.add_hyperparameters([
            loss, learning_rate, n_estimators, max_depth, min_samples_split,
            min_samples_leaf, subsample, max_features, alpha])

        super().__init__(base.Primitive.GradientBoostingRegressor, configuration_space)


class XGradientBoostingRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        objective = Constant("objective", "reg:squarederror")
        max_depth = UniformIntegerHyperparameter("max_depth", lower=1, upper=10, default_value=3)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        subsample = UniformFloatHyperparameter("subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
        min_child_weight = UniformIntegerHyperparameter("min_child_weight", lower=1, upper=20,
                                                        default_value=1, log=False)

        configuration_space.add_hyperparameters([objective, max_depth, learning_rate,
                                                 n_estimators, subsample, min_child_weight])

        super().__init__(base.Primitive.XGradientBoostingRegressor, configuration_space)


class ARDRegressionRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_iter = UnParametrizedHyperparameter("n_iter", value=300)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)
        alpha_1 = UniformFloatHyperparameter("alpha_1", 1e-10, 1e-3, default_value=1e-6)
        alpha_2 = UniformFloatHyperparameter("alpha_2", 1e-10, 1e-3, default_value=1e-6, log=True)
        lambda_1 = UniformFloatHyperparameter("lambda_1", 1e-10, 1e-3, default_value=1e-6, log=True)
        lambda_2 = UniformFloatHyperparameter("lambda_2", 1e-10, 1e-3, default_value=1e-6, log=True)
        threshold_lambda = UniformFloatHyperparameter("threshold_lambda", 1e3, 1e5, default_value=1e4, log=True)

        configuration_space.add_hyperparameters([
            n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, threshold_lambda])

        super().__init__(base.Primitive.ARDRegression, configuration_space)


class DecisionTreeRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        criterion = CategoricalHyperparameter("criterion", ['mse', 'friedman_mse', 'mae'], default_value="mse")
        max_depth = UniformFloatHyperparameter('max_depth', 0., 2., default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)

        configuration_space.add_hyperparameters([
            criterion, max_depth, min_samples_split, min_samples_leaf])

        super().__init__(base.Primitive.DecisionTreeRegressor, configuration_space)


class LGBMRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        num_leaves = UniformIntegerHyperparameter("num_leaves", lower=2, upper=128, default_value=31)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=-1, upper=10, default_value=-1)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)

        configuration_space.add_hyperparameters([num_leaves, max_depth, learning_rate, n_estimators])

        super().__init__(base.Primitive.LGBMRegressor, configuration_space)


class RuleFitRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        tree_size = UniformIntegerHyperparameter("tree_size", lower=2, upper=16, default_value=4)
        max_rules = UniformIntegerHyperparameter("max_rules", lower=1000, upper=10000, default_value=2000)

        configuration_space.add_hyperparameters([tree_size, max_rules])

        super().__init__(base.Primitive.RuleFit, configuration_space)


class CatBoostRegressorRule(RegressionRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        num_leaves = UniformIntegerHyperparameter("num_leaves", lower=2, upper=128, default_value=31)
        max_depth = UniformIntegerHyperparameter("max_depth", lower=-1, upper=10, default_value=-1)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)

        configuration_space.add_hyperparameters([num_leaves, max_depth, learning_rate, n_estimators])

        super().__init__(base.Primitive.CatBoostRegressor, configuration_space)


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    rule_executor.register_parameter_rule(SVRRule())
    rule_executor.register_parameter_rule(LinearSVRRule())
    rule_executor.register_parameter_rule(LinearRegressionRule())
    rule_executor.register_parameter_rule(RidgeRegressionRule())
    rule_executor.register_parameter_rule(SGDRegressorRule())
    rule_executor.register_parameter_rule(RandomForestRegressorRule())
    rule_executor.register_parameter_rule(GaussianProcessRegressorRule())
    rule_executor.register_parameter_rule(AdaBoostRegressorRule())
    rule_executor.register_parameter_rule(KNeighborsRegressorRule())
    rule_executor.register_parameter_rule(BaggingRegressorRule())
    rule_executor.register_parameter_rule(ExtraTreesRegressorRule())
    rule_executor.register_parameter_rule(GradientBoostingRegressorRule())
    rule_executor.register_parameter_rule(XGradientBoostingRegressorRule())
    rule_executor.register_parameter_rule(ARDRegressionRule())
    rule_executor.register_parameter_rule(DecisionTreeRegressorRule())
    rule_executor.register_parameter_rule(LGBMRegressorRule())
    rule_executor.register_parameter_rule(RuleFitRule())
    rule_executor.register_parameter_rule(CatBoostRegressorRule())
