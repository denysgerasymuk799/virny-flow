# pylint: disable=missing-docstring
"""Rules for adding preprocessing primitives, e.g., scaler, encoder, etc."""
from pyarrow.types import is_boolean, is_string, is_integer, is_floating, is_timestamp

from alpine_meadow.common import TaskKeyword
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.search_space import PipelineArmStep
from alpine_meadow.core.rule.primitive.rule import PrimitiveRule


class EncoderRule(PrimitiveRule):

    def predicate(self, task):
        for field in task.dataset.schema:
            if field.name not in task.target_columns:
                if is_boolean(field.type) or is_string(field.type):
                    return True

        return False

    def apply(self, search_space):
        encoder_classes_all = [(base.Primitive.LabelEncoder,)]
        if not search_space.task.config.explainable_feature_processing:
            encoder_classes_all.append((base.Primitive.LabelEncoder, base.Primitive.OneHotEncoder))

        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # find inputs
            input_step_index = None
            for step_index, step in enumerate(pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'attributes' in step.tags:
                    input_step_index = step_index
                    break
            if input_step_index is None:
                continue

            # add encoders
            for encoder_classes in encoder_classes_all:
                # new pipeline arm
                new_pipeline_arm = pipeline_arm.copy()
                new_pipeline_arm.primitive_rules.add(self)

                # extract boolean/categorical columns
                step_inputs = {
                    'inputs': f'steps.{input_step_index}'
                }
                step = PipelineArmStep(base.Primitive.ExtractColumnsByNames, inputs=step_inputs)
                step.constant_parameters['names'] = []
                for field in search_space.task.dataset.schema:
                    if field.name not in search_space.task.target_columns:
                        if is_boolean(field.type) or is_string(field.type):
                            step.constant_parameters['names'].append(field.name)
                new_pipeline_arm.add_step(step)

                for encoder_class in encoder_classes:
                    step_inputs = {
                        'inputs': f'steps.{len(new_pipeline_arm.steps) - 1}'
                    }
                    step = PipelineArmStep(encoder_class, inputs=step_inputs)
                    if encoder_class == base.Primitive.OneHotEncoder:
                        # step.constant_parameters['sparse'] = False
                        # step.constant_parameters['handle_unknown'] = 'ignore'
                        step.constant_parameters['minimum_fraction'] = 0.01
                    new_pipeline_arm.add_step(step)

                new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


class TimestampRule(PrimitiveRule):

    def predicate(self, task):
        for field in task.dataset.schema:
            if field.name not in task.target_columns:
                if is_timestamp(field.type):
                    return True

        return False

    def apply(self, search_space):
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # find inputs
            input_step_index = None
            for step_index, step in enumerate(pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'attributes' in step.tags:
                    input_step_index = step_index
                    break
            if input_step_index is None:
                continue

            # extract numerical columns
            step_inputs = {
                'inputs': f'steps.{input_step_index}'
            }
            step = PipelineArmStep(base.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = []
            for field in search_space.task.dataset.schema:
                if field.name not in search_space.task.target_columns:
                    if is_timestamp(field.type):
                        step.constant_parameters['names'].append(field.name)
            pipeline_arm.add_step(step)

            # add timestamp converter
            step_inputs = {
                'inputs': f'steps.{len(pipeline_arm.steps) - 1}'
            }
            step = PipelineArmStep(base.Primitive.TimestampConverter, inputs=step_inputs)
            pipeline_arm.add_step(step)

            # add imputer
            step_inputs = {
                'inputs': f'steps.{len(pipeline_arm.steps) - 1}'
            }
            step = PipelineArmStep(base.Primitive.Imputer, inputs=step_inputs)
            pipeline_arm.add_step(step)

            pipeline_arm.primitive_rules.add(self)


class ImputerRule(PrimitiveRule):

    def predicate(self, task):
        for field in task.dataset.schema:
            if field.name not in task.target_columns:
                if is_integer(field.type) or is_floating(field.type):
                    return True

        return False

    def apply(self, search_space):
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # find inputs
            input_step_index = None
            for step_index, step in enumerate(pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'attributes' in step.tags:
                    input_step_index = step_index
                    break
            if input_step_index is None:
                continue

            # extract numerical columns
            step_inputs = {
                'inputs': f'steps.{input_step_index}'
            }
            step = PipelineArmStep(base.Primitive.ExtractColumnsByNames, inputs=step_inputs)
            step.constant_parameters['names'] = []
            for field in search_space.task.dataset.schema:
                if field.name not in search_space.task.target_columns:
                    if is_integer(field.type) or is_floating(field.type):
                        step.constant_parameters['names'].append(field.name)
            pipeline_arm.add_step(step)

            # add imputer
            step_inputs = {
                'inputs': f'steps.{len(pipeline_arm.steps) - 1}'
            }
            step = PipelineArmStep(base.Primitive.Imputer, inputs=step_inputs)
            pipeline_arm.add_step(step)

            pipeline_arm.primitive_rules.add(self)


class ScalerRule(PrimitiveRule):

    def predicate(self, task):
        for field in task.dataset.schema:
            if field.name not in task.target_columns:
                if is_integer(field.type) or is_floating(field.type):
                    return True

        return False

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # find inputs
            input_step_index = None
            for step_index, step in reversed(list(enumerate(pipeline_arm.steps))):
                if step.primitive == base.Primitive.Imputer:
                    input_step_index = step_index
                    continue

            if input_step_index is None:
                continue

            for scaler_class in [base.Primitive.MinMaxScaler, base.Primitive.StandardScaler,
                                 base.Primitive.RobustScaler, base.Primitive.Normalizer]:
                # new pipeline arm
                new_pipeline_arm = pipeline_arm.copy()
                new_pipeline_arm.primitive_rules.add(self)

                # step inputs
                step_inputs = {
                    'inputs': f'steps.{input_step_index}'
                }

                # add scaler
                step = PipelineArmStep(scaler_class, inputs=step_inputs)
                new_pipeline_arm.add_step(step)

                new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


class FeatureSelectionRule(PrimitiveRule):

    def predicate(self, task):
        # only works for classification and regression tasks
        task_type = task.type
        if task_type not in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]:
            return False
        attributes_num = len(task.dataset.schema) - len(task.target_columns)
        if attributes_num < 3:
            return False
        if task.config.explainable_feature_processing:
            return False

        return True

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # new pipeline arm
            new_base_pipeline_arm = pipeline_arm.copy()
            new_base_pipeline_arm.primitive_rules.add(self)

            # exclude some rules
            new_base_pipeline_arm.excluding_rules.add(FeatureReductionRule)
            new_base_pipeline_arm.excluding_rules.add(FeatureGenerationRule)

            # find inputs
            input_step_indexes = []
            output_step_index = None
            for step_index, step in enumerate(new_base_pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'targets' in step.tags:
                    output_step_index = step_index
                    continue

                no_output = True
                for step_index2 in range(step_index + 1, len(new_base_pipeline_arm.steps)):
                    if f'steps.{step_index}' in new_base_pipeline_arm.steps[step_index2].inputs.values():
                        no_output = False
                        break

                if no_output:
                    input_step_indexes.append(step_index)
            if not input_step_indexes or output_step_index is None:
                continue

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
                    new_base_pipeline_arm.add_step(step)
                    last_input_step_index = len(new_base_pipeline_arm.steps) - 1
                input_step_index = last_input_step_index

            # add utils
            step_inputs = {
                'inputs': f'steps.{input_step_index}',
                'outputs': f'steps.{output_step_index}'
            }
            for primitive_class in [base.Primitive.VarianceThreshold,
                                    base.Primitive.SelectPercentile]:  # [base.Primitive.GenericUnivariateSelect]:
                # new pipeline arm
                new_pipeline_arm = new_base_pipeline_arm.copy()

                # add scaler
                step = PipelineArmStep(primitive_class, inputs=step_inputs)
                new_pipeline_arm.add_step(step)

                new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


class FeatureReductionRule(PrimitiveRule):

    def predicate(self, task):
        # only works for classification and regression tasks
        task_type = task.type
        if task_type not in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]:
            return False
        attributes_num = len(task.dataset.schema) - len(task.target_columns)
        if attributes_num < 3:
            return False
        if task.config.explainable_feature_processing:
            return False

        return True

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # new pipeline arm
            new_base_pipeline_arm = pipeline_arm.copy()
            new_base_pipeline_arm.primitive_rules.add(self)

            # exclude some rules
            new_base_pipeline_arm.excluding_rules.add(FeatureSelectionRule)
            new_base_pipeline_arm.excluding_rules.add(FeatureGenerationRule)

            # find inputs
            input_step_indexes = []
            output_step_index = None
            for step_index, step in enumerate(new_base_pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'targets' in step.tags:
                    output_step_index = step_index
                    continue

                no_output = True
                for step_index2 in range(step_index + 1, len(new_base_pipeline_arm.steps)):
                    if str(step_index) in new_base_pipeline_arm.steps[step_index2].inputs.get('inputs', ''):
                        no_output = False
                        break

                if no_output:
                    input_step_indexes.append(step_index)
            if not input_step_indexes or output_step_index is None:
                continue

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
                    new_base_pipeline_arm.add_step(step)
                    last_input_step_index = len(new_base_pipeline_arm.steps) - 1
                input_step_index = last_input_step_index

            # add utils
            for primitive_class in [base.Primitive.PCA, base.Primitive.KernelPCA,
                                    base.Primitive.FastICA, base.Primitive.TruncatedSVD]:
                # new pipeline arm
                new_pipeline_arm = new_base_pipeline_arm.copy()

                # step inputs
                step_inputs = {
                    'inputs': f'steps.{input_step_index}'
                }

                # add scaler
                step = PipelineArmStep(primitive_class, inputs=step_inputs)
                new_pipeline_arm.add_step(step)

                new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


class FeatureGenerationRule(PrimitiveRule):

    def predicate(self, task):
        # only works for classification and regression tasks
        task_type = task.type
        if task_type not in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]:
            return False

        attributes_num = len(task.dataset.schema) - len(task.target_columns)
        if attributes_num >= 20:
            return False
        if task.config.explainable_feature_processing:
            return False

        return True

    def apply(self, search_space):
        new_pipeline_arms = []
        for pipeline_arm in search_space.pipeline_arms:
            # apply only once
            if self in pipeline_arm.primitive_rules:
                continue
            if pipeline_arm.allowing_rules is not None and not any(map(lambda rule: isinstance(self, rule),
                                                                       pipeline_arm.allowing_rules)):
                continue
            if any(map(lambda rule: isinstance(self, rule), pipeline_arm.excluding_rules)):
                continue

            # new pipeline arm
            new_base_pipeline_arm = pipeline_arm.copy()
            new_base_pipeline_arm.primitive_rules.add(self)

            # exclude feature selection rule
            new_base_pipeline_arm.excluding_rules.add(FeatureSelectionRule)
            new_base_pipeline_arm.excluding_rules.add(FeatureReductionRule)

            # find inputs
            input_step_index = None
            output_step_index = None
            for step_index, step in enumerate(new_base_pipeline_arm.steps):
                if step.primitive == base.Primitive.ExtractColumnsByNames and 'targets' in step.tags:
                    output_step_index = step_index
                    continue

                if step.primitive in [base.Primitive.Imputer, base.Primitive.MinMaxScaler,
                                      base.Primitive.StandardScaler, base.Primitive.RobustScaler]:
                    input_step_index = step_index
                    continue
            if input_step_index is None or output_step_index is None:
                continue

            # add utils
            for primitive_class in [base.Primitive.PolynomialFeatures, base.Primitive.FeatureAgglomeration,
                                    base.Primitive.RBFSampler]:
                # new pipeline arm
                new_pipeline_arm = new_base_pipeline_arm.copy()

                # step inputs
                step_inputs = {
                    'inputs': f'steps.{input_step_index}'
                }
                if primitive_class in [base.Primitive.PolynomialFeatures]:
                    step_inputs['outputs'] = f'steps.{output_step_index}'

                # add scaler
                step = PipelineArmStep(primitive_class, inputs=step_inputs)
                new_pipeline_arm.add_step(step)

                new_pipeline_arms.append(new_pipeline_arm)

        for pipeline_arm in new_pipeline_arms:
            search_space.add_pipeline_arm(pipeline_arm)


def register_rules(rule_executor):
    if not rule_executor.config.enable_feature_processing:
        return

    rule_executor.register_primitive_rule(EncoderRule())
    rule_executor.register_primitive_rule(TimestampRule())
    rule_executor.register_primitive_rule(ImputerRule())
    rule_executor.register_primitive_rule(ScalerRule())
    rule_executor.register_primitive_rule(FeatureSelectionRule())
    rule_executor.register_primitive_rule(FeatureReductionRule())
    rule_executor.register_primitive_rule(FeatureGenerationRule())
