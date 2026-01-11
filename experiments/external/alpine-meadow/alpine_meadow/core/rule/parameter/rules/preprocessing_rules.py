# pylint: disable=missing-docstring
"""Rules for preprocessing primitives, including scaler, encoder, etc."""
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition
# from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenInClause, \
#     ForbiddenEqualsClause

from alpine_meadow.common import TaskKeyword
from alpine_meadow.common.proto import pipeline_pb2 as base
from ..rule import ParameterRule


class PreprocessingRule(ParameterRule):
    """
    The base class for parameter rules of preprocessing primitives.
    """

    def __init__(self, primitive, configuration_space):
        self.primitive = primitive
        self.configuration_space = configuration_space

    @property
    def primitive(self):
        return self.__estimator

    @primitive.setter
    def primitive(self, primitive):
        self.__estimator = primitive

    def predicate(self, task, step):
        return step.primitive == self.primitive

    def apply(self, task, step):
        step.configuration_space = self.configuration_space


class ImputerRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # strategy = CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent"], default_value="mean")
        strategy = CategoricalHyperparameter("strategy", ["mean"], default_value="mean")

        configuration_space.add_hyperparameter(strategy)

        super().__init__(base.Primitive.Imputer, configuration_space)


class LabelEncoderRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        super().__init__(base.Primitive.LabelEncoder, configuration_space)


class OneHotEncoderRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # categorical_features = Constant('categorical_features', 'all')
        # minimum_fraction = UniformFloatHyperparameter("minimum_fraction",
        # lower=.001, upper=0.5, default_value=0.01, log=True)

        # configuration_space.add_hyperparameter(categorical_features)

        super().__init__(base.Primitive.OneHotEncoder, configuration_space)


class SelectPercentileRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        percentile = UniformIntegerHyperparameter("percentile", 90, 99, default_value=95)

        configuration_space.add_hyperparameter(percentile)

        super().__init__(base.Primitive.SelectPercentile, configuration_space)

        self.updated = False

    def apply(self, task, step):
        if not self.updated:
            # for classification
            from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

            score_func_classification = CategoricalHyperparameter(
                "score_func", choices=[chi2, f_classif, mutual_info_classif], default_value=f_classif)

            # for regression
            from sklearn.feature_selection import f_regression, mutual_info_regression

            score_func_regression = CategoricalHyperparameter(
                "score_func", choices=[f_regression, mutual_info_regression], default_value=f_regression)

            if task.type == TaskKeyword.Value('CLASSIFICATION'):
                self.configuration_space.add_hyperparameter(score_func_classification)
            elif task.type == TaskKeyword.Value('REGRESSION'):
                self.configuration_space.add_hyperparameter(score_func_regression)

            self.updated = True

        step.configuration_space = self.configuration_space


class GenericUnivariateSelectRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        alpha = UniformFloatHyperparameter(name="alpha", lower=0.01, upper=0.5, default_value=0.1)
        mode = CategoricalHyperparameter(name="mode", choices=['fpr', 'fdr', 'fwe'], default_value='fpr')

        configuration_space.add_hyperparameters([alpha, mode])

        super().__init__(base.Primitive.GenericUnivariateSelect, configuration_space)

        self.updated = False

    def apply(self, task, step):
        if not self.updated:
            # for classification
            # score_func_classification = CategoricalHyperparameter("score_func",
            # choices=["f_classif", "mutual_info_classif"], default_value="f_classif")
            score_func_classification = Constant("score_func", "f_classif")

            # for regression
            # score_func_regression = CategoricalHyperparameter("score_func",
            # choices=["f_regression", "mutual_info_regression"], default_value="f_regression")
            score_func_regression = Constant("score_func", "f_regression")

            if task.type == TaskKeyword.Value('CLASSIFICATION'):
                self.configuration_space.add_hyperparameter(score_func_classification)
            elif task.type == TaskKeyword.Value('REGRESSION'):
                self.configuration_space.add_hyperparameter(score_func_regression)

            self.updated = True

        step.configuration_space = self.configuration_space


class VarianceThresholdRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        threshold = UniformFloatHyperparameter(name="threshold", lower=0.0, upper=5.0, default_value=0.0)

        configuration_space.add_hyperparameter(threshold)

        super().__init__(base.Primitive.VarianceThreshold, configuration_space)


class PCARule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        n_components = UniformFloatHyperparameter("n_components", 0.5, 0.9999, default_value=0.9999)
        # n_components = UniformIntegerHyperparameter("n_components", 10, 100, default_value=20)
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)

        configuration_space.add_hyperparameters([n_components, whiten])

        super().__init__(base.Primitive.PCA, configuration_space)


class KernelPCARule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # n_components = UniformIntegerHyperparameter("n_components", 10, 2000, default_value=100)
        kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=1.0)
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        remove_zero_eig = CategoricalHyperparameter("remove_zero_eig", [True], True)

        configuration_space.add_hyperparameters([kernel, gamma, degree, coef0, remove_zero_eig])

        # conditions
        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        # coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        # gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])

        configuration_space.add_condition(degree_depends_on_poly)

        super().__init__(base.Primitive.KernelPCA, configuration_space)


class TruncatedSVDRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        super().__init__(base.Primitive.TruncatedSVD, configuration_space)

        self.updated = False

    def apply(self, task, step):
        if not self.updated:
            # hyper parameters
            attributes_num = len(task.dataset.schema) - len(task.target_columns)
            components_num = min(attributes_num - 1, int(0.05 * task.dataset.num_instances))
            components_num = max(components_num, 2)
            default_value = min(10, components_num)
            n_components = UniformIntegerHyperparameter("n_components", 1, components_num,
                                                        default_value=default_value)
            self.configuration_space.add_hyperparameter(n_components)
            self.updated = True

        step.configuration_space = self.configuration_space


class FastICARule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        # n_components = UniformFloatHyperparameter("n_components", 0.5, 0.9999, default_value=0.9999)
        algorithm = CategoricalHyperparameter('algorithm', ['parallel', 'deflation'], 'parallel')
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)
        fun = CategoricalHyperparameter('fun', ['logcosh', 'exp', 'cube'], 'logcosh')

        configuration_space.add_hyperparameters([algorithm, whiten, fun])

        super().__init__(base.Primitive.FastICA, configuration_space)


class PolynomialFeaturesRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only", [False, True], False)
        include_bias = CategoricalHyperparameter("include_bias", [True, False], True)

        configuration_space.add_hyperparameters([degree, interaction_only, include_bias])

        super().__init__(base.Primitive.PolynomialFeatures, configuration_space)


class FeatureAgglomerationRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        affinity = CategoricalHyperparameter("affinity", ["euclidean", "manhattan", "cosine"], "euclidean")
        linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"], "ward")

        configuration_space.add_hyperparameters([affinity, linkage])

        # forbiddens
        # affinity_and_linkage = ForbiddenAndConjunction(
        #     ForbiddenInClause(affinity, ["manhattan", "cosine"]),
        #     ForbiddenEqualsClause(linkage, "ward"))
        # configuration_space.add_forbidden_clause(affinity_and_linkage)

        super().__init__(base.Primitive.FeatureAgglomeration, configuration_space)

        self.updated = False

    def apply(self, task, step):
        if not self.updated:
            # hyper parameters
            attributes_num = len(task.dataset.schema) - len(task.target_columns)
            if attributes_num > 2 and task.dataset.num_instances > 100:
                components_num = min(attributes_num - 1, int(0.05 * task.dataset.num_instances))
                default_value = min(10, components_num)
                n_clusters = UniformIntegerHyperparameter("n_clusters", 1, components_num,
                                                          default_value=default_value)
                self.configuration_space.add_hyperparameter(n_clusters)

            self.updated = True

        step.configuration_space = self.configuration_space


class RBFSamplerRule(PreprocessingRule):

    def __init__(self):
        # configuration space
        configuration_space = ConfigurationSpace()

        # hyper parameters
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, default_value=1.0, log=True)
        n_components = UniformIntegerHyperparameter("n_components", 50, 10000, default_value=100, log=True)

        configuration_space.add_hyperparameters([gamma, n_components])

        super().__init__(base.Primitive.RBFSampler, configuration_space)


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    rule_executor.register_parameter_rule(ImputerRule())
    rule_executor.register_parameter_rule(LabelEncoderRule())
    rule_executor.register_parameter_rule(OneHotEncoderRule())
    rule_executor.register_parameter_rule(SelectPercentileRule())
    rule_executor.register_parameter_rule(GenericUnivariateSelectRule())
    rule_executor.register_parameter_rule(VarianceThresholdRule())
    rule_executor.register_parameter_rule(PCARule())
    rule_executor.register_parameter_rule(KernelPCARule())
    rule_executor.register_parameter_rule(TruncatedSVDRule())
    rule_executor.register_parameter_rule(FastICARule())
    rule_executor.register_parameter_rule(PolynomialFeaturesRule())
    rule_executor.register_parameter_rule(FeatureAgglomerationRule())
    rule_executor.register_parameter_rule(RBFSamplerRule())
