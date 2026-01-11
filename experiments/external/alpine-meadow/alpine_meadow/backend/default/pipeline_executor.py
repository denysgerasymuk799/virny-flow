# pylint: disable=cell-var-from-loop
"""Implementation for the default pipeline executor."""

import json
import collections

import pandas as pd
import numpy as np

from alpine_meadow.common import PerformanceMetric
from alpine_meadow.backend.base import BasePipelineExecutor, TrainResult, TestResult, ScoreResult
from alpine_meadow.utils import ignore_warnings, AMException
from .primitive import get_primitive_from_step, get_primitive_name_from_step, get_method_arguments


class DefaultPipelineExecutor(BasePipelineExecutor):
    """
    Default implementation for the pipeline executor running inside
    Alpine Meadow.
    """

    def __init__(self, pipeline, metrics, backend=None):
        super().__init__(pipeline, metrics)
        self._pipeline_desc = pipeline.to_pipeline_desc()
        self._primitives = []
        self._backend = backend
        self.trained_dataset_ids = []
        self.scored_dataset_ids = []

    @property
    def primitives(self):
        return self._primitives

    @primitives.setter
    def primitives(self, primitives):
        self._primitives = primitives

    @property
    def model(self):
        for primitive in self.primitives:
            if hasattr(primitive.primitive, 'classes_'):
                return primitive
        return None

    @ignore_warnings
    def train(self, datasets, **kwargs) -> TrainResult:
        for dataset in datasets:
            self.trained_dataset_ids.append(dataset.id)
        inputs = [dataset.to_raw_data() for dataset in datasets]
        outputs, metrics = self._run(inputs, **kwargs)
        return TrainResult(outputs=outputs, metrics=metrics)

    @ignore_warnings
    def test(self, datasets, **kwargs) -> TestResult:
        inputs = [dataset.to_raw_data() for dataset in datasets]
        outputs, metrics = self._run(inputs, False, **kwargs)
        return TestResult(metrics=metrics, outputs=outputs)

    @ignore_warnings
    def test_proba(self, datasets, **kwargs) -> TestResult:  # pylint: disable=unused-argument
        inputs = [dataset.to_raw_data() for dataset in datasets]
        outputs, metrics = self._run(inputs, False, True)
        return TestResult(metrics=metrics, outputs=outputs)

    @ignore_warnings
    def score(self, datasets, targets, **kwargs) -> ScoreResult:
        for dataset in datasets:
            self.scored_dataset_ids.append(dataset.id)

        from alpine_meadow.common.metric import get_score

        inputs = [dataset.to_raw_data() for dataset in datasets]
        outputs, metrics = self._run(inputs, False, **kwargs)
        scores = []
        for metric in self.metrics:
            values = outputs.values
            if 'ROC_AUC' in PerformanceMetric.Name(metric):
                if len(values.shape) > 1 and values.shape[1] > 1:
                    truth = pd.DataFrame()
                    for label in list(outputs.columns):
                        truth[label] = np.array(targets[0].values == label).flatten()
                else:
                    truth = targets[0].values
            else:
                truth = targets[0].values
                if len(values.shape) > 1 and values.shape[1] > 1:
                    values = outputs.idxmax(axis=1).values

            score = get_score(metric, truth, values, **kwargs)
            self.add_score(metric, score)
            scores.append(score)
        return ScoreResult(scores=scores, metrics=metrics)

    @ignore_warnings
    def _run(self, inputs, is_train=True, produce_proba=False, **kwargs):  # pylint: disable=unused-argument
        """
        Train or test the pipeline and return the trained primitives, outputs and internal metrics.
        :param inputs:
        :param is_train:
        :param produce_proba:
        :return:
        """

        import time
        from collections import defaultdict

        pipeline = self.pipeline
        primitives = self._primitives
        metrics = defaultdict(float)
        metrics['step_runs'] = []
        outputs = []
        for step_index, step in enumerate(pipeline.steps):
            # step run
            step_run = {'index': step_index}

            # check later used inputs
            all_later_inputs = set()
            for later_step_index in range(step_index, len(pipeline.steps)):
                later_step = pipeline.steps[later_step_index]
                for argument, input_value in later_step.inputs.items():
                    all_later_inputs.add(input_value)

            # remove unused inputs
            if not any(map(lambda input_value: 'inputs' in input_value, all_later_inputs)):
                inputs = None

            # remove unused outputs
            for former_step_index in range(0, step_index):
                if outputs[former_step_index] is not None and not any(
                        map(lambda all_later_input: f'steps.{former_step_index}' in all_later_input,
                            all_later_inputs)):
                    outputs[former_step_index] = None

            # get primitive
            if len(primitives) <= step_index:
                if is_train:
                    primitive = get_primitive_from_step(step)
                    primitives.append(primitive)
                else:
                    # sometimes we only want to get the intermediate output
                    break
            else:
                primitive = primitives[step_index]
            primitive_name = get_primitive_name_from_step(step)
            step_run['primitive'] = primitive_name

            # get primitive arguments
            primitive_arguments = {}
            for argument, input_value in step.inputs.items():
                if input_value.startswith('inputs.'):
                    input_index = int(input_value.split('.')[1])
                    primitive_arguments[argument] = inputs[input_index]
                else:
                    if not input_value.startswith('steps.'):
                        raise AMException(f"Unknown input: {input_value}")
                    tokens = input_value.split('.')
                    output_index = int(tokens[1])
                    primitive_arguments[argument] = outputs[output_index]

            # train
            if is_train:
                start = time.perf_counter()
                # prepare train arguments
                training_arguments_metadata = get_method_arguments(primitive.set_training_data)
                training_arguments = {}
                for argument, value in primitive_arguments.items():
                    if argument in training_arguments_metadata:
                        training_arguments[argument] = value

                # train
                primitive.set_training_data(**training_arguments)
                primitive.fit()
                metrics[
                    f'primitive.{primitive_name}.train_time'] += time.perf_counter() - start
                metrics[f'primitive.{primitive_name}.train_count'] += 1
                step_run['fit_time'] = time.perf_counter() - start
            else:
                step_run['fit_time'] = None

            # prepare produce arguments
            start = time.perf_counter()
            produce_arguments_metadata = get_method_arguments(primitive.produce)
            produce_arguments = {}

            for argument, value in primitive_arguments.items():
                if argument in produce_arguments_metadata:
                    produce_arguments[argument] = value

            # produce
            if step.outputs:
                produce_method = step.outputs[0]
                output = getattr(primitive, produce_method)(**produce_arguments)
            else:
                if produce_proba and hasattr(primitive, "produce_proba"):
                    try:
                        output = primitive.produce_proba(**produce_arguments)
                    except:  # noqa: E722  # pylint:disable=bare-except
                        output = primitive.produce(**produce_arguments)
                else:
                    output = primitive.produce(**produce_arguments)
            outputs.append(output)

            # metrics
            metrics[
                f'primitive.{primitive_name}.inference_time'] += time.perf_counter() - start
            metrics[f'primitive.{primitive_name}.inference_count'] += 1
            step_run['produce_time'] = time.perf_counter() - start

            # add step run
            metrics['step_runs'].append(step_run)

        return outputs[-1], metrics

    def copy(self):
        another = DefaultPipelineExecutor(self.pipeline, self.metrics, backend=self._backend)
        another._primitives = self.primitives  # pylint: disable=protected-access
        return another

    def export(self):
        """
        Export the pipeline as a Python script
        """

        import os
        from datetime import datetime
        from .primitive import LOGICAL_TO_PHYSICAL_TABLE, unpickle_parameters, base

        comment = f'# This script is exported by Alpine Meadow at {datetime.now()}'

        imports = set()
        code = ''
        for step_index, step in enumerate(self.pipeline.steps):
            # set up inputs
            step_inputs = []
            ordered_inputs = collections.OrderedDict(sorted(step.inputs.items()))
            for input_value in ordered_inputs.values():
                if input_value.startswith('inputs.'):
                    input_index = int(input_value[7:])
                    step_inputs.append(f'inputs[{input_index}]')
                else:
                    if not input_value.startswith('steps.'):
                        raise AMException(f"Unknown input: {input_value}")
                    output_index = int(input_value[6:])
                    step_inputs.append(f'output_{output_index}')

            # parameters
            parameters = unpickle_parameters(step.primitive.parameters)

            if step.primitive.name == base.Primitive.ExtractColumnsByNames:
                if len(step_inputs) != 1:
                    raise AMException(f"ExtractColumnsByNames expects exactly one input: {len(step_inputs)}")

                code += f"step_inputs = [{', '.join(step_inputs)}]"
                code += '\n'
                code += 'output_{} = step_inputs[0][[{}]]'.format(  # pylint: disable=consider-using-f-string
                    step_index,
                    ', '.join(map('"{}"'.format, parameters['names'])))  # pylint: disable=consider-using-f-string
                code += '\n'
                code += '\n'
                continue

            if step.primitive.name == base.Primitive.HorizontalConcat:
                if len(step_inputs) < 2:
                    raise AMException(f"HorizontalConcat expects more than one inputs: {len(step_inputs)}")

                code += f"step_inputs = [{', '.join(step_inputs)}]"
                code += '\n'
                code += f'output_{step_index} = pd.concat(step_inputs, axis=1, ignore_index=True)'
                code += '\n'
                code += '\n'
                continue

            # import
            method = 'transform'
            if step.primitive.name == base.Primitive.LabelEncoder:
                path = 'sklearn.preprocessing.OrdinalEncoder'
                method = 'transform'
            elif step.primitive.name == base.Primitive.OneHotEncoder:
                path = 'sklearn.preprocessing.OneHotEncoder'
                parameters = {**parameters, 'sparse': False}
                method = 'transform'
            else:
                path = LOGICAL_TO_PHYSICAL_TABLE[step.primitive.name]
                if getattr(self._primitives[step_index].primitive, "predict", None):
                    method = 'predict'
            module_path = '.'.join(path.split('.')[:-1])
            class_name = path.split('.')[-1]
            imports.add(f'from {module_path} import {class_name}')

            # construct class
            primitive = f'step_{step_index}'
            code += f"parameters = json.loads('{json.dumps(parameters)}')"
            code += '\n'
            code += f'{primitive} = {class_name}(**parameters)'
            code += '\n'

            # fit
            code += f"step_inputs = [{', '.join(step_inputs)}]"
            code += '\n'
            code += f'{primitive}.fit(*step_inputs)'
            code += '\n'

            # transform
            code += f'output_{step_index} = {primitive}.{method}(step_inputs[0])'
            code += '\n'
            code += f'output_{step_index} = pd.DataFrame(output_{step_index})'
            code += '\n'
            code += '\n'

        code += '# print output'
        code += '\n'
        code += f'output = pd.DataFrame(output_{len(self.pipeline.steps) - 1})'
        code += '\n'
        code += 'print(output)'
        code += '\n'

        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pipeline_script.template')) as f:
            pipeline_script_template = f.read()

        return pipeline_script_template.format(comment=comment, imports='\n'.join(imports), code=code)
