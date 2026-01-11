"""Rules for randomly combining two pipeline arms into one pipeline arm to generate
data-specific pipeline arms."""
import random
import copy

import numpy as np

from alpine_meadow.core.search_space import PipelineArm, PipelineArmStep
from alpine_meadow.common.proto import pipeline_pb2 as base
from ..rule import MutationRule


class ColumnsMutationRule(MutationRule):
    """
    This rule randomly merge two pipeline arms together, i.e., run each pipeline arm on a disjoint set of columns and
    concatenate their results together
    """

    def apply(self, search_space):
        pipeline_arms = search_space.pipeline_arms
        if len(pipeline_arms) == 1:
            return pipeline_arms[0]

        while True:
            left_pipeline_arm = np.random.choice(pipeline_arms)
            right_pipeline_arm = np.random.choice(pipeline_arms)

            if left_pipeline_arm == right_pipeline_arm:
                continue
            break

        first_left_step_index = None
        first_right_step_index = None
        for step_index, step in enumerate(left_pipeline_arm.steps):
            if step.primitive == base.Primitive.ExtractColumnsByNames and 'attributes' in step.tags:
                break
        for step_index, step in enumerate(right_pipeline_arm.steps):
            if step.primitive == base.Primitive.ExtractColumnsByNames and 'attributes' in step.tags:
                first_right_step_index = step_index
                break

        if first_left_step_index is None or first_right_step_index is None:
            return left_pipeline_arm

        # new pipeline arm
        new_pipeline_arm = PipelineArm(left_pipeline_arm.pipeline_history)
        new_pipeline_arm.search_space = left_pipeline_arm.search_space
        new_pipeline_arm.primitive_rules = left_pipeline_arm.primitive_rules.union(right_pipeline_arm.primitive_rules)
        new_pipeline_arm.parameter_rules = left_pipeline_arm.parameter_rules.union(right_pipeline_arm.parameter_rules)
        if left_pipeline_arm.allowing_rules is None:
            new_pipeline_arm.allowing_rules = copy.copy(right_pipeline_arm.allowing_rules)
        elif right_pipeline_arm.allowing_rules is None:
            new_pipeline_arm.allowing_rules = copy.copy(left_pipeline_arm.allowing_rules)
        else:
            new_pipeline_arm.allowing_rules = left_pipeline_arm.allowing_rules.union(right_pipeline_arm.allowing_rules)
        new_pipeline_arm.excluding_rules = left_pipeline_arm.excluding_rules.union(right_pipeline_arm.excluding_rules)
        new_pipeline_arm.complete = left_pipeline_arm.complete and right_pipeline_arm.complete

        # split columns
        columns_names = left_pipeline_arm.steps[first_left_step_index].constant_parameters['use_columns_names']
        left_columns_names = []
        right_columns_names = []
        for column_name in columns_names:
            if random.random() <= 0.5:
                left_columns_names.append(column_name)
            else:
                right_columns_names.append(column_name)

        # merge left steps (all but last step)
        left_steps = copy.deepcopy(left_pipeline_arm.steps)
        left_steps[first_left_step_index].constant_parameters['use_columns_names'] = list(left_columns_names)
        for left_step in left_steps[:-1]:
            new_pipeline_arm.add_step(left_step)
        last_left_step_index = len(new_pipeline_arm.steps) - 1

        # merge right steps (all but last step)
        right_step_index_delta = last_left_step_index + 1
        right_steps = copy.deepcopy(right_pipeline_arm.steps)
        for right_step in right_steps[:-1]:
            for input_key in right_step.inputs:
                if right_step.inputs[input_key].startswith('steps.'):
                    right_step_index = int(right_step.inputs[input_key][6:]) + right_step_index_delta
                    right_step.inputs[input_key] = f'steps.{right_step_index}'

            new_pipeline_arm.add_step(right_step)

        # merge outputs from left and right
        step_inputs = {
            'left': f'steps.{last_left_step_index}',
            'right': f'steps.{len(new_pipeline_arm.steps) - 1}',
        }
        step = PipelineArmStep(base.Primitive.HorizontalConcat, inputs=step_inputs)
        new_pipeline_arm.add_step(step)

        # use left pipeline's predictor
        last_step = left_steps[-1]
        last_step.inputs['inputs'] = f'steps.{len(new_pipeline_arm.steps) - 1}'
        # fix names of parameters
        primitive_name = str(base.Primitive.Name.Name(last_step.primitive))
        prefix = primitive_name + '_'
        if last_step.configuration_space:
            for hp in last_step.configuration_space.get_hyperparameters():
                if hp.name.startswith(prefix):
                    hp.name = hp.name[len(prefix):]
        new_pipeline_arm.add_step(last_step)

        new_pipeline_arm.create_configuration_space()

        return new_pipeline_arm


def register_rules(rule_executor):
    rule_executor.register_mutation_rule(ColumnsMutationRule())
