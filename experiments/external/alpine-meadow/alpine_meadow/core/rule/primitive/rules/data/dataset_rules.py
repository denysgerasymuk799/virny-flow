"""Primitive rules for reading different types of datasets,
e.g., tabular, image."""

from alpine_meadow.common import Task
from alpine_meadow.core.search_space import SearchSpace, PipelineArmStep
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule
from alpine_meadow.common.proto import pipeline_pb2
from alpine_meadow.common import DatasetType


class DatasetRule(PrimitiveRule):  # pylint: disable=abstract-method
    """
    The base class for dataset rules, which read the data (either tabular or non-tabular) from the inputs
    """


class FeatureEngineeringRule(DatasetRule):
    """
    Apply feature engineering over the dataset.
    """

    def predicate(self, task: Task):
        return task.dataset.type == DatasetType.Value('TABULAR') and hasattr(task, '_engineered_features')

    def apply(self, search_space: SearchSpace):
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(
                    map(lambda rule: isinstance(self, rule), pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            pipeline_arm.primitive_rules.add(self)

            # feature engineering step
            step_inputs = {
                'inputs': 'inputs.0'
            }
            step = PipelineArmStep(pipeline_pb2.Primitive.FeatureEngineering, inputs=step_inputs)
            best_features = search_space.task._engineered_features  # pylint: disable=protected-access
            step.constant_parameters['features'] = best_features
            pipeline_arm.add_step(step)

            # extract non-target columns
            step_inputs = {
                'inputs': f'steps.{len(pipeline_arm.steps) - 1}'
            }
            step = PipelineArmStep(pipeline_pb2.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = []
            for field in search_space.task.dataset.schema:
                if field.name not in search_space.task.target_columns:
                    step.constant_parameters['names'].append(field.name)
            step.tags.append('attributes')
            pipeline_arm.add_step(step)

            # extract target columns
            step_inputs = {
                'inputs': 'inputs.0'
            }
            step = PipelineArmStep(pipeline_pb2.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = search_space.task.target_columns
            step.tags.append('targets')
            pipeline_arm.add_step(step)

            # exclude other rules
            pipeline_arm.excluding_rules.add(DatasetRule)


class TabularDatasetRule(DatasetRule):
    """
    Rule for reading from tabular dataset.
    """

    def predicate(self, task: Task):
        return task.dataset.type == DatasetType.Value('TABULAR')

    def apply(self, search_space: SearchSpace):
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            pipeline_arm.primitive_rules.add(self)

            # extract non-target columns
            step_inputs = {
                'inputs': 'inputs.0'
            }
            step = PipelineArmStep(pipeline_pb2.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = []
            for field in search_space.task.dataset.schema:
                if field.name not in search_space.task.target_columns:
                    step.constant_parameters['names'].append(field.name)
            step.tags.append('attributes')
            pipeline_arm.add_step(step)

            # extract target columns
            step_inputs = {
                'inputs': 'inputs.0'
            }
            step = PipelineArmStep(pipeline_pb2.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = search_space.task.target_columns
            step.tags.append('targets')
            pipeline_arm.add_step(step)

            # exclude other rules
            pipeline_arm.excluding_rules.add(DatasetRule)


def register_rules(rule_executor):
    rule_executor.register_primitive_rule(FeatureEngineeringRule())
    rule_executor.register_primitive_rule(TabularDatasetRule())
