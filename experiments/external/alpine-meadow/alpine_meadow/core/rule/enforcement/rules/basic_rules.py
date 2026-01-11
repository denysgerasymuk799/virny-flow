"""Some basic enforcement rules."""
from alpine_meadow.common import TaskKeyword
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.rule.primitive.rules.data.dataset_rules import TabularDatasetRule
from alpine_meadow.utils import AMException
from ..rule import EnforcementRule


class RemoveEmptyPipelineRule(EnforcementRule):
    """
    This rule removes empty pipeline arms
    """

    @staticmethod
    def enforce(task, pipeline_arm):
        if not pipeline_arm.steps:
            return False

        return True


class CheckEncoderRule(EnforcementRule):
    """
    This rule removes pipeline arms without encoders
    """

    @staticmethod
    def predicate(task):
        from alpine_meadow.core.rule.primitive.rules.feature import preprocessing_rules

        return preprocessing_rules.EncoderRule().predicate(task)

    @staticmethod
    def enforce(task, pipeline_arm):
        # only check tabular dataset
        if not any(map(lambda rule: isinstance(rule, (TabularDatasetRule,)), pipeline_arm.primitive_rules)):
            return True

        from alpine_meadow.core.rule.primitive.rules.feature import preprocessing_rules

        return any(map(lambda primitive_rule: isinstance(primitive_rule, preprocessing_rules.EncoderRule),
                       pipeline_arm.primitive_rules))


class CheckImputerRule(EnforcementRule):
    """
    This rule removes pipeline arms without imputers
    """

    @staticmethod
    def predicate(task):
        from alpine_meadow.core.rule.primitive.rules.feature import preprocessing_rules

        return preprocessing_rules.ImputerRule().predicate(task)

    @staticmethod
    def enforce(task, pipeline_arm):
        # only check tabular dataset
        if not any(map(lambda rule: isinstance(rule, (TabularDatasetRule,)), pipeline_arm.primitive_rules)):
            return True

        from alpine_meadow.core.rule.primitive.rules.feature import preprocessing_rules

        return any(map(lambda primitive_rule: isinstance(primitive_rule, preprocessing_rules.ImputerRule),
                       pipeline_arm.primitive_rules))


class CheckClassificationRule(EnforcementRule):
    """
    This rule removes pipeline arms without classifiers for a classification task
    """

    @staticmethod
    def enforce(task, pipeline_arm):
        from alpine_meadow.core.rule.primitive.rules.model import classification_rules

        keywords = task.keywords
        if TaskKeyword.Value('CLASSIFICATION') in keywords:
            return any(map(lambda primitive_rule: isinstance(primitive_rule, classification_rules.ClassificationRule),
                           pipeline_arm.primitive_rules))

        return True


class CheckRegressionRule(EnforcementRule):
    """
    This rule removes pipeline arms without regressors for a regression task
    """

    @staticmethod
    def enforce(task, pipeline_arm):
        from alpine_meadow.core.rule.primitive.rules.model import regression_rules

        keywords = task.keywords
        if TaskKeyword.Value('REGRESSION') in keywords:
            return any(map(lambda primitive_rule: isinstance(primitive_rule, regression_rules.RegressionRule),
                           pipeline_arm.primitive_rules))

        return True


class DisableErrorPronePrimitivesRule(EnforcementRule):
    """
    This rule removes pipeline arms with utils that are likely to cause errors
    """

    @staticmethod
    def enforce(task, pipeline_arm):
        error_prone_primitives = {base.Primitive.FastICA, base.Primitive.SelectPercentile, base.Primitive.KernelPCA,
                                  base.Primitive.VarianceThreshold, base.Primitive.FeatureAgglomeration,
                                  base.Primitive.GradientBoostingClassifier,
                                  base.Primitive.GradientBoostingRegressor}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        if error_prone_primitives.intersection(pipeline_arm_primitives):
            return False

        return True


class DisableWeakPerformancePrimitivesRule(EnforcementRule):
    """
    This rule removes pipeline arms with utils that are considerably weak
    """

    @staticmethod
    def enforce(task, pipeline_arm):
        error_prone_primitives = {base.Primitive.LinearDiscriminantAnalysis, base.Primitive.LinearRegression}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        if error_prone_primitives.intersection(pipeline_arm_primitives):
            return False

        return True


class CheckStandardScalerForSVMRule(EnforcementRule):
    """
    We want to use standard scaler for SVMs because they provide more stable inputs to SVMs.
    """

    @staticmethod
    def predicate(task):
        from alpine_meadow.core.rule.primitive.rules.feature import preprocessing_rules

        return preprocessing_rules.ScalerRule().predicate(task)

    @staticmethod
    def enforce(task, pipeline_arm):
        svm_primitives = {base.Primitive.SVC, base.Primitive.SVR, base.Primitive.LinearSVC, base.Primitive.LinearSVR}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        if svm_primitives.intersection(pipeline_arm_primitives):
            return base.Primitive.StandardScaler in pipeline_arm_primitives

        return True


class DisableNonLinearSVMForLargeDatasetRule(EnforcementRule):
    """
    Since non-linear SVM is too slow for large datasets, we will disable them.
    """

    @staticmethod
    def predicate(task):
        if task.meta_features is not None:
            instances_num = task.meta_features['NumberOfInstances'].value
            features_num = task.meta_features['NumberOfFeatures'].value
            return instances_num > 10000 or features_num > 20

        return False

    @staticmethod
    def enforce(task, pipeline_arm):
        non_linear_svm_primitives = {base.Primitive.SVC, base.Primitive.SVR}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return not non_linear_svm_primitives.intersection(pipeline_arm_primitives)


class DisableKNNPrimitivesForLargeDatasetRule(EnforcementRule):
    """
    Since some KNN primitives are too slow for large datasets, we will disable them.
    """

    @staticmethod
    def predicate(task):
        if task.meta_features is not None:
            instances_num = task.meta_features['NumberOfInstances'].value
            features_num = task.meta_features['NumberOfFeatures'].value
            return instances_num > 1000 or features_num > 50

        return False

    @staticmethod
    def enforce(task, pipeline_arm):
        knn_primitives = {
            base.Primitive.KNeighborsClassifier, base.Primitive.KNeighborsRegressor}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return not knn_primitives.intersection(pipeline_arm_primitives)


class FastPrimitivesForLargeDatasetRule(EnforcementRule):
    """
    We only use fast primitives for large datasets.
    """

    @staticmethod
    def predicate(task):
        if task.meta_features is not None:
            instances_num = task.meta_features['NumberOfInstances'].value
            features_num = task.meta_features['NumberOfFeatures'].value
            return instances_num > 1000000 or features_num > 100

        return False

    @staticmethod
    def enforce(task, pipeline_arm):
        fast_primitives = {
            base.Primitive.RandomForestClassifier, base.Primitive.RandomForestRegressor,
            base.Primitive.LGBMClassifier, base.Primitive.LGBMRegressor,
            base.Primitive.XGradientBoostingClassifier, base.Primitive.XGradientBoostingRegressor}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return fast_primitives.intersection(pipeline_arm_primitives)


class PredictProbabilityRule(EnforcementRule):
    """
    For classification problems, if we want to get the predicted probabilities, we will only
    use those classifiers capable of doing that.
    """

    @staticmethod
    def predicate(task):
        keywords = task.keywords
        return TaskKeyword.Value('CLASSIFICATION') in keywords and task.config.predict_proba

    @staticmethod
    def enforce(task, pipeline_arm):
        predict_prob_primitives = {
            base.Primitive.LogisticRegression, base.Primitive.RandomForestClassifier,
            base.Primitive.GaussianNB, base.Primitive.AdaBoostClassifier,
            base.Primitive.BaggingClassifier, base.Primitive.ExtraTreesClassifier,
            base.Primitive.GradientBoostingClassifier, base.Primitive.KNeighborsClassifier,
            base.Primitive.XGradientBoostingClassifier, base.Primitive.LinearDiscriminantAnalysis,
            base.Primitive.QuadraticDiscriminantAnalysis, base.Primitive.DecisionTreeClassifier,
            base.Primitive.LGBMClassifier}
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return predict_prob_primitives.intersection(pipeline_arm_primitives)


class OnlyTreeModelsRule(EnforcementRule):
    """
    The rule constrains that we only use tree-based models.
    """

    @staticmethod
    def predicate(task):
        if not task.config.only_tree_models:
            return False

        keywords = task.keywords
        return TaskKeyword.Value('CLASSIFICATION') in keywords or TaskKeyword.Value('REGRESSION') in keywords

    @staticmethod
    def enforce(task, pipeline_arm):
        keywords = task.keywords
        if TaskKeyword.Value('CLASSIFICATION') in keywords:
            tree_primitives = {
                base.Primitive.RandomForestClassifier,
                # base.Primitive.ExtraTreesClassifier,
                # base.Primitive.GradientBoostingClassifier,
                base.Primitive.XGradientBoostingClassifier,
                # base.Primitive.DecisionTreeClassifier,
                base.Primitive.LGBMClassifier,
                base.Primitive.CatBoostClassifier
            }
        else:
            if TaskKeyword.Value('REGRESSION') not in keywords:
                raise AMException("Currently we only support CLASSIFICATION and REGRESSION")
            tree_primitives = {
                base.Primitive.RandomForestRegressor,
                # base.Primitive.ExtraTreesRegressor,
                # base.Primitive.GradientBoostingRegressor,
                base.Primitive.XGradientBoostingRegressor,
                # base.Primitive.DecisionTreeRegressor,
                base.Primitive.LGBMRegressor,
                base.Primitive.CatBoostRegressor
            }
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return tree_primitives.intersection(pipeline_arm_primitives)


class DisableSlowModelsForHighCardinalityTargetRule(EnforcementRule):
    """
    The rule removes the slow classifiers when the target column has a high cardinality.
    """

    @staticmethod
    def predicate(task) -> bool:
        if TaskKeyword.Value('CLASSIFICATION') not in task.keywords:
            return False
        meta_features = task.meta_features
        if meta_features is not None and 'NumberOfClasses' in meta_features.keys()\
                and meta_features['NumberOfClasses'].value > 100:
            return True
        return False

    @staticmethod
    def enforce(task, pipeline_arm):
        fast_primitives = {
            base.Primitive.RandomForestClassifier,
            # base.Primitive.XGradientBoostingClassifier,
            # base.Primitive.LGBMClassifier,
            # base.Primitive.CatBoostClassifier
        }
        pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
        return fast_primitives.intersection(pipeline_arm_primitives)


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    rule_executor.register_enforcement_rule(RemoveEmptyPipelineRule)
    rule_executor.register_enforcement_rule(CheckEncoderRule)
    rule_executor.register_enforcement_rule(CheckClassificationRule)
    rule_executor.register_enforcement_rule(CheckRegressionRule)
    rule_executor.register_enforcement_rule(DisableErrorPronePrimitivesRule)
    rule_executor.register_enforcement_rule(DisableWeakPerformancePrimitivesRule)
    rule_executor.register_enforcement_rule(CheckStandardScalerForSVMRule)
    rule_executor.register_enforcement_rule(DisableNonLinearSVMForLargeDatasetRule)
    rule_executor.register_enforcement_rule(DisableKNNPrimitivesForLargeDatasetRule)
    rule_executor.register_enforcement_rule(FastPrimitivesForLargeDatasetRule)
    rule_executor.register_enforcement_rule(PredictProbabilityRule)
    rule_executor.register_enforcement_rule(OnlyTreeModelsRule)
    rule_executor.register_enforcement_rule(DisableSlowModelsForHighCardinalityTargetRule)
