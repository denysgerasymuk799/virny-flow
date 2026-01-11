"""Rules for using dummy classifiers or regressors as the baseline."""
from alpine_meadow.core.search_space import PipelineArmStep
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.common import TaskKeyword
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule
from alpine_meadow.utils import AMException


class DummyBaselineRule(PrimitiveRule):
    """
    This rule creates a new pipeline which is able to generate outputs for any task.
    """

    def predicate(self, task):
        task_type = task.type
        return task_type in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]

    def apply(self, search_space):
        from alpine_meadow.core.rule.primitive.rules.model.classification_rules \
            import ClassificationRule
        from alpine_meadow.core.rule.primitive.rules.model.regression_rules \
            import RegressionRule

        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if any(map(lambda primitive_rule: isinstance(primitive_rule, ClassificationRule),
                       pipeline_arm.primitive_rules)):
                continue
            if any(map(lambda primitive_rule: isinstance(primitive_rule, RegressionRule),
                       pipeline_arm.primitive_rules)):
                continue
            if pipeline_arm.allowing_rules is not None and not any(
                    map(lambda rule: isinstance(self, rule), pipeline_arm.allowing_rules)):
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
            keywords = search_space.task.keywords
            if TaskKeyword.Value('CLASSIFICATION') in keywords:
                step = PipelineArmStep(base.Primitive.DummyClassifier, inputs=step_inputs)
            else:
                if TaskKeyword.Value('REGRESSION') not in keywords:
                    raise AMException("Currently we only support CLASSIFICATION and REGRESSION")
                step = PipelineArmStep(base.Primitive.DummyRegressor, inputs=step_inputs)
            new_pipeline_arm.add_step(step)

        return new_pipeline_arms


def register_rules(rule_executor):
    rule_executor.register_primitive_rule(DummyBaselineRule())
