"""Rules for classifiers."""
from alpine_meadow.common import TaskKeyword
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.search_space import PipelineArmStep
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule


class ClassificationRule(PrimitiveRule):
    """
    Base rule class for adding classifier into the search space.
    """

    def __init__(self, estimator, constant_parameters=None):
        self.estimator = estimator
        self.constant_parameters = constant_parameters

    @property
    def estimator(self):
        return self.__estimator

    @estimator.setter
    def estimator(self, estimator):
        self.__estimator = estimator

    def predicate(self, task):
        task_type = task.type
        return task_type in [TaskKeyword.Value('CLASSIFICATION')]

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if any(map(lambda primitive_rule: isinstance(primitive_rule, ClassificationRule),
                       pipeline_arm.primitive_rules)):
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # find inputs and outputs
            input_step_indexes = []
            output_step_index = []
            for step_index, step in enumerate(pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'targets' in step.tags:
                    output_step_index = step_index
                else:
                    no_output = True
                    for step_index2 in range(step_index + 1, len(pipeline_arm.steps)):
                        if f'steps.{step_index}' in pipeline_arm.steps[step_index2].inputs.values():
                            no_output = False
                            break

                    if no_output:
                        input_step_indexes.append(step_index)
            if not input_step_indexes or output_step_index is None:
                continue

            # new pipeline arm
            new_pipeline_arm = pipeline_arm.copy()
            new_pipeline_arm.primitive_rules.add(self)
            new_pipeline_arms.append(new_pipeline_arm)

            # concatenate inputs
            if len(input_step_indexes) == 1:
                input_step_index = input_step_indexes[0]
            else:
                last_input_step_index = input_step_indexes[0]
                for input_step_index in input_step_indexes[1:]:
                    step_inputs = {
                        'left': f'steps.{last_input_step_index}',
                        'right': f'steps.{input_step_index}',
                    }
                    step = PipelineArmStep(base.Primitive.HorizontalConcat, inputs=step_inputs)
                    new_pipeline_arm.add_step(step)
                    last_input_step_index = len(new_pipeline_arm.steps) - 1
                input_step_index = last_input_step_index

            # add estimator
            step_inputs = {
                'inputs': f'steps.{input_step_index}',
                'outputs': f'steps.{output_step_index}'
            }
            step = PipelineArmStep(self.estimator, inputs=step_inputs)
            if self.constant_parameters:
                step.constant_parameters = dict(self.constant_parameters)
            new_pipeline_arm.add_step(step)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


class SVCRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.SVC)


class LinearSVCRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.LinearSVC)


class LogisticRegressionRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.LogisticRegression)


class SGDClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.SGDClassifier)


class RandomForestClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.RandomForestClassifier)


class GaussianNBRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.GaussianNB)


class AdaBoostClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.AdaBoostClassifier)


class KNeighborsClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.KNeighborsClassifier)


class BaggingClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.BaggingClassifier)


class ExtraTreesClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.ExtraTreesClassifier)


class GradientBoostingClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.GradientBoostingClassifier)


class XGradientBoostingClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.XGradientBoostingClassifier, {
            'n_jobs': 1
        })


class LinearDiscriminantAnalysisRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.LinearDiscriminantAnalysis)


class QuadraticDiscriminantAnalysisRule(ClassificationRule):
    """
    Don't use QuadraticDiscriminantAnalysis where there are too many classes since it
    will be super slow.
    """

    def __init__(self):
        super().__init__(base.Primitive.QuadraticDiscriminantAnalysis)

    def predicate(self, task):
        if super().predicate(task):
            meta_features = task.meta_features
            if meta_features is None:
                return False
            if 'ClassProbabilityMin' in meta_features.keys() and meta_features['ClassProbabilityMin'].value < 0.01:
                return False

            return True
        return False


class DecisionTreeClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.DecisionTreeClassifier)


class LGBMClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.LGBMClassifier, {
            'n_jobs': 1
        })


class CatBoostClassifierRule(ClassificationRule):

    def __init__(self):
        super().__init__(base.Primitive.CatBoostClassifier, {
            'verbose': False
        })


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    # rule_executor.register_primitive_rule(SVCRule())
    # rule_executor.register_primitive_rule(LinearSVCRule())
    rule_executor.register_primitive_rule(LogisticRegressionRule())
    rule_executor.register_primitive_rule(SGDClassifierRule())
    rule_executor.register_primitive_rule(RandomForestClassifierRule())
    rule_executor.register_primitive_rule(GaussianNBRule())
    rule_executor.register_primitive_rule(AdaBoostClassifierRule())
    rule_executor.register_primitive_rule(BaggingClassifierRule())
    rule_executor.register_primitive_rule(ExtraTreesClassifierRule())
    rule_executor.register_primitive_rule(GradientBoostingClassifierRule())
    rule_executor.register_primitive_rule(KNeighborsClassifierRule())
    rule_executor.register_primitive_rule(XGradientBoostingClassifierRule())
    rule_executor.register_primitive_rule(LinearDiscriminantAnalysisRule())
    rule_executor.register_primitive_rule(QuadraticDiscriminantAnalysisRule())
    rule_executor.register_primitive_rule(DecisionTreeClassifierRule())
    rule_executor.register_primitive_rule(LGBMClassifierRule())
    # rule_executor.register_primitive_rule(CatBoostClassifierRule())
