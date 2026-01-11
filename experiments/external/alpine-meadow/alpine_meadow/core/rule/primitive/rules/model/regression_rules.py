# pylint: disable=missing-class-docstring
"""Rules for regressors."""
from alpine_meadow.common import TaskKeyword
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.search_space import PipelineArmStep
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule


class RegressionRule(PrimitiveRule):
    """
    Base rule class for adding regressor into the search space.
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
        return task_type in [TaskKeyword.Value('REGRESSION')]

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if any(map(lambda primitive_rule: isinstance(primitive_rule, RegressionRule),
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


class SVRRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.SVR)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class LinearSVRRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.LinearSVR)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class LinearRegressionRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.LinearRegression)


class RidgeRegressionRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.Ridge)


class SGDRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.SGDRegressor)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class RandomForestRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.RandomForestRegressor)


class GaussianProcessRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.GaussianProcessRegressor)


class AdaBoostRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.AdaBoostRegressor)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class KNeighborsRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.KNeighborsRegressor)


class BaggingRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.BaggingRegressor)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class ExtraTreesRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.ExtraTreesRegressor)


class GradientBoostingRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.GradientBoostingRegressor)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class XGradientBoostingRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.XGradientBoostingRegressor, {
            'n_jobs': 1
        })

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class ARDRegressionRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.ARDRegression)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class DecisionTreeRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.DecisionTreeRegressor)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class LGBMRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.LGBMRegressor, {
            'n_jobs': 1
        })

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class RuleFitRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.RuleFit)

    def predicate(self, task):
        if super().predicate(task) and len(task.target_columns) == 1:
            return True

        return False


class CatBoostRegressorRule(RegressionRule):

    def __init__(self):
        super().__init__(base.Primitive.CatBoostRegressor, {
            'verbose': False
        })


def register_rules(rule_executor):
    """
    Register all rules.
    :param rule_executor:
    :return:
    """

    # rule_executor.register_primitive_rule(SVRRule())
    # rule_executor.register_primitive_rule(LinearSVRRule())
    rule_executor.register_primitive_rule(LinearRegressionRule())
    rule_executor.register_primitive_rule(RidgeRegressionRule())
    rule_executor.register_primitive_rule(SGDRegressorRule())
    rule_executor.register_primitive_rule(RandomForestRegressorRule())
    rule_executor.register_primitive_rule(GaussianProcessRegressorRule())
    rule_executor.register_primitive_rule(AdaBoostRegressorRule())
    rule_executor.register_primitive_rule(BaggingRegressorRule())
    rule_executor.register_primitive_rule(ExtraTreesRegressorRule())
    rule_executor.register_primitive_rule(GradientBoostingRegressorRule())
    rule_executor.register_primitive_rule(KNeighborsRegressorRule())
    rule_executor.register_primitive_rule(XGradientBoostingRegressorRule())
    rule_executor.register_primitive_rule(ARDRegressionRule())
    rule_executor.register_primitive_rule(DecisionTreeRegressorRule())
    rule_executor.register_primitive_rule(LGBMRegressorRule())
    # rule_executor.register_primitive_rule(RuleFitRule())
    # rule_executor.register_primitive_rule(CatBoostRegressorRule())
