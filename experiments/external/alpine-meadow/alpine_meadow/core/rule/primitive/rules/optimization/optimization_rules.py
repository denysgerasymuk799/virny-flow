"""Rules for optimizing the pipline arms, e.g., adding constant hyper-parameters."""
import copy

from alpine_meadow.common import TaskKeyword, PerformanceMetric
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.search_space import PipelineArmStep
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule


class WarmStartRule(PrimitiveRule):
    """
    Rule for enabling warm-start for some primitives.
    """

    def predicate(self, task):
        return True

    def apply(self, search_space):
        warm_start_primitives = [
            base.Primitive.LogisticRegression,
            base.Primitive.SGDClassifier, base.Primitive.SGDRegressor,
            base.Primitive.RandomForestClassifier, base.Primitive.RandomForestRegressor,
            base.Primitive.BaggingClassifier, base.Primitive.BaggingRegressor,
            base.Primitive.ExtraTreesClassifier, base.Primitive.ExtraTreesRegressor,
            base.Primitive.GradientBoostingClassifier, base.Primitive.GradientBoostingRegressor]
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue

            # set warm start to True for applicable utils
            for step in pipeline_arm.steps:
                if step.primitive in warm_start_primitives:
                    step.constant_parameters['warm_start'] = True


class ThresholdTuningRule(PrimitiveRule):
    """
    Rule for adding threshold tuning primitive.
    """

    def predicate(self, task):
        config = task.config
        if not config.predict_proba or not config.enable_threshold_tuning:
            return False

        if task.type != TaskKeyword.Value('CLASSIFICATION'):
            return False
        if 'F1' not in PerformanceMetric.Name(task.metrics[0]):
            return False

        return True

    def apply(self, search_space):
        from alpine_meadow.core.rule.primitive.rules.model import classification_rules

        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            if 'outputs' not in pipeline_arm.steps[-1].inputs:
                continue

            if not any(map(lambda primitive_rule:
                           isinstance(primitive_rule, classification_rules.ClassificationRule),
                           pipeline_arm.primitive_rules)):
                continue

            new_pipeline_arm = pipeline_arm.copy()

            # update previous step
            new_pipeline_arm.steps[-1] = copy.copy(new_pipeline_arm.steps[-1])
            new_pipeline_arm.steps[-1].outputs = copy.copy(new_pipeline_arm.steps[-1].outputs)
            new_pipeline_arm.steps[-1].outputs.append('produce_proba')

            # add new step
            step_inputs = {
                'inputs': f'steps.{len(new_pipeline_arm.steps) - 1}.produce_proba',
                'outputs': new_pipeline_arm.steps[-1].inputs['outputs']
            }
            step = PipelineArmStep(base.Primitive.ThresholdingPrimitive, inputs=step_inputs)
            new_pipeline_arm.add_step(step)

            # add new pipeline arm
            new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


def register_rules(rule_executor):
    # rule_executor.register_primitive_rule(WarmStartRule())
    rule_executor.register_primitive_rule(ThresholdTuningRule())
