"""Alpine Meadow pipeline arm, which is a logical pipeline plan consisting a family of
pipelines sharing the same structure (i.e., primitives and how they are connected) with different
hyper-parameters."""
import copy
import pickle
import tempfile
import os
import datetime
import time

import numpy as np
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from smac.runhistory.runhistory import RunHistory
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant

from alpine_meadow.common import Pipeline
from alpine_meadow.common.proto import pipeline_pb2
from alpine_meadow.utils.performance import time_calls


class PipelineArmStep:
    """
    A single step in the pipeline arm
    """

    def __init__(self, primitive, inputs):
        self.primitive = primitive
        self.primitive_name = str(pipeline_pb2.Primitive.Name.Name(self.primitive))
        self.inputs = inputs
        self.outputs = []
        self.configuration_space = None
        self.constant_parameters = {}
        self.tags = []

    def __repr__(self):
        return f'[primitive={self.primitive}, inputs={self.inputs}, constant_parameters={self.constant_parameters}]'

    def get_configuration_count(self):
        """
        Get the total number of hyper-parameter configurations for this step/primitive
        """

        count = 1
        if self.configuration_space is None:
            return count

        for hp in self.configuration_space.get_hyperparameters():
            if isinstance(hp, Constant):
                continue

            if isinstance(hp, CategoricalHyperparameter):
                count *= len(hp.choices)
                continue

            return None

        return count


EMPTY_CONFIGURATION_SPACE = ConfigurationSpace()


class PipelineArm:
    """
    The class for pipeline arm, which consists of all the steps and a configuration space of hyper-parameters.
    By combining all these steps and a configuration of hyper-parameters, we get a concrete pipeline
    """

    def __init__(self, pipeline_history):
        # steps
        self.steps = []

        # rules
        self.primitive_rules = set()
        self.parameter_rules = set()
        self.allowing_rules = None
        self.excluding_rules = set()
        self.complete = False

        # configurations
        self.configuration_space = None
        self.run_history = RunHistory()
        self.external_run_history = RunHistory()
        self.starting_configurations = None

        # pipeline history
        self.pipeline_history = pipeline_history
        self.quality_mean = None
        self.quality_std = None
        self.time_mean = None
        self.external_quality_mean = None
        self.external_quality_std = None
        self.search_space = None

    def add_step(self, step):
        self.steps.append(step)

    def get_primitives_strs(self):
        primitives_strs = []
        for step in self.steps:
            primitive_name = str(pipeline_pb2.Primitive.Name.Name(step.primitive))
            primitives_strs.append(primitive_name)
        return primitives_strs

    def get_unique_primitives(self):
        primitives = []
        for step in self.steps:
            if step.primitive not in primitives:
                primitives.append(step.primitive)
        return primitives

    def get_unique_primitives_strs(self):
        primitives_strs = []
        for step in self.steps:
            primitive_name = str(pipeline_pb2.Primitive.Name.Name(step.primitive))
            if primitive_name not in primitives_strs:
                primitives_strs.append(primitive_name)
        return primitives_strs

    def get_unique_tunable_primitives_strs(self):
        primitives_strs = []
        for step in self.steps:
            if step.configuration_space is None:
                continue

            primitive_name = str(pipeline_pb2.Primitive.Name.Name(step.primitive))
            if primitive_name not in primitives_strs:
                primitives_strs.append(primitive_name)
        return primitives_strs

    @time_calls
    def get_next_pipeline(self, use_bayesian_optimization=False):
        return self.get_next_pipelines(use_bayesian_optimization=use_bayesian_optimization, pipelines_num=1)[0]

    @time_calls
    def get_next_pipelines(self, use_bayesian_optimization=False, pipelines_num=1):
        """
        Get next k promising pipelines from this pipeline arm.
        It uses `_get_next_configurations` to get the next k configurations and assemble they
        as pipelines.
        :param use_bayesian_optimization:
        :param pipelines_num:
        :return:
        """

        if self.configuration_space is None:
            self.create_configuration_space()

        configurations = self._get_next_configurations(use_bayesian_optimization=use_bayesian_optimization,
                                                       configurations_num=pipelines_num)
        pipelines = []
        for configuration, tags in configurations:
            pipeline = self.get_pipeline_from_configuration(configuration)
            pipeline.tags = {**pipeline.tags, **tags}
            pipelines.append(pipeline)

        return pipelines

    def get_default_pipeline(self):
        configuration = self.configuration_space.get_default_configuration()
        pipeline = self.get_pipeline_from_configuration(configuration)
        pipeline.tags['hyperparameters'] = 'default'

        return pipeline

    def get_pipelines_count(self):
        """
        Get the total number of pipelines from this pipeline arm, and this is decided by
        the hyper-parameter space.
        If there are continuous ranges, then the number will be
        `None` (for infinite).
        :return:
        """

        pipelines_count = 1
        for step in self.steps:
            count = step.get_configuration_count()
            if count is None:
                pipelines_count = None
                break
            pipelines_count *= count

        return pipelines_count

    def _get_next_configurations(self, use_bayesian_optimization=False, configurations_num=1):
        """
        Get the next k promising hyper-parameter configurations.
        :param use_bayesian_optimization:
        :param configurations_num:
        :return:
        """

        run_history = self.run_history
        external_run_history = self.external_run_history
        configurations = []

        # initialize starting configurations
        if self.starting_configurations is None:
            self.starting_configurations = [
                (self.configuration_space.get_default_configuration(),
                 {'hyperparameters': 'default'})]
            sorted_configs = sorted(external_run_history.config_ids.keys(), key=external_run_history.get_cost)
            starting_configurations_num = self.search_space.config.starting_configurations_from_history_num
            for configuration in list(sorted_configs)[:starting_configurations_num]:
                self.starting_configurations.append((configuration, {'hyperparameters': 'meta-learning'}))

        # use starting configurations first
        if self.starting_configurations:
            configurations.extend(self.starting_configurations[:configurations_num])
            self.starting_configurations = self.starting_configurations[configurations_num:]

        if len(configurations) == configurations_num:
            return configurations
        configurations_num -= len(configurations)

        if not use_bayesian_optimization or self.get_pipelines_count() == 1 or run_history.empty():
            random_configurations = self.configuration_space.sample_configuration(configurations_num + 1)
            for configuration in random_configurations[:configurations_num]:
                configurations.append((configuration, {'hyperparameters': 'random'}))
            self.search_space.increase_num_searched_pipelines(len(configurations))

            return configurations

        # init bayesian optimization solver
        scenario = Scenario({
            "run_obj": "quality",
            "cs": self.configuration_space,
            "deterministic": True,
            "output_dir": os.path.join(
                tempfile.gettempdir(),
                f"smac3-output_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')}")
        })
        smac = SMAC(scenario=scenario, runhistory=run_history,
                    rng=np.random.RandomState())
        solver = smac.solver

        # find new configurations
        @time_calls
        def get_bo_configurations(solver, expected_configurations_num):
            challengers = solver.epm_chooser.choose_next()
            configurations_count = 0
            self.search_space.config.logger.debug(f'{len(challengers)} configurations found in Bayesian Optimization, '
                                                  f'top {expected_configurations_num} will be used')
            self.search_space.increase_num_searched_pipelines(len(challengers))
            for challenger in challengers:
                yield challenger
                configurations_count += 1
                if configurations_count == expected_configurations_num:
                    break

        extra_configurations = get_bo_configurations(solver, configurations_num)
        for configuration in extra_configurations:
            configurations.append((configuration, {'hyperparameters': 'bayesian'}))
            self.search_space.increase_num_searched_pipelines(1)

        return configurations

    def copy(self):
        """
        Return a shallow copy of this pipeline arm.
        :return:
        """

        pipeline_arm = PipelineArm(self.pipeline_history)
        # pipeline_arm.steps = copy.deepcopy(self.steps)
        pipeline_arm.steps = copy.copy(self.steps)
        pipeline_arm.primitive_rules = copy.copy(self.primitive_rules)
        pipeline_arm.parameter_rules = copy.copy(self.parameter_rules)
        pipeline_arm.allowing_rules = copy.copy(self.allowing_rules)
        pipeline_arm.excluding_rules = copy.copy(self.excluding_rules)
        pipeline_arm.complete = self.complete

        return pipeline_arm

    @time_calls
    def create_configuration_space(self):
        """
        Create a hyper-parameter configuration space for this pipeline arm,
        including the hyper-parameters of all primitives.
        :return:
        """

        configuration_space = copy.deepcopy(EMPTY_CONFIGURATION_SPACE)

        primitives_names = set()
        for step in self.steps:
            primitive_name = str(pipeline_pb2.Primitive.Name.Name(step.primitive))
            if step.configuration_space is not None:
                if str(primitive_name) not in primitives_names:
                    try:
                        configuration_space.add_configuration_space(
                            configuration_space=step.configuration_space,
                            prefix=str(primitive_name), delimiter='_')
                        primitives_names.add(str(primitive_name))
                    except BaseException:  # pylint: disable=broad-except
                        # raise
                        pass

        self.configuration_space = configuration_space

    def get_pipeline_from_configuration(self, configuration):
        """
        Assemble a pipeline from the given hyper-parameter configuration.
        :param configuration:
        :return:
        """

        pipeline = Pipeline(pipeline_arm=self, configuration=configuration)
        self.pipeline_history.save_pipeline(pipeline)

        def get_parameters(step):
            primitive_name = str(pipeline_pb2.Primitive.Name.Name(step.primitive))
            prefix = primitive_name + '_'
            parameters = {}
            for parameter_name in configuration:
                if parameter_name.startswith(prefix):
                    new_parameter_name = parameter_name[len(prefix):]
                    parameters[new_parameter_name] = pickle.dumps(configuration[parameter_name])

            for parameter_name in step.constant_parameters:
                parameters[parameter_name] = pickle.dumps(step.constant_parameters[parameter_name])

            return parameters

        # get steps
        for step in self.steps:
            parameters = get_parameters(step)
            primitive = pipeline_pb2.Primitive(name=step.primitive, parameters=parameters)
            pipeline_step = pipeline_pb2.Step(primitive=primitive, inputs=step.inputs, outputs=step.outputs)
            pipeline.steps.append(pipeline_step)

        pipeline.tags['model'] = str(pipeline_pb2.Primitive.Name.Name(self.steps[-1].primitive))

        return pipeline

    def compute_metrics(self):
        """
        Re-compute the metrics by iterating through all the pipelines.
        :return:
        """

        scores = []
        times = []
        for pipeline in self.pipeline_history.get_pipelines_by_pipeline_arm(self):
            metrics = pipeline.metrics
            if metrics.score:
                scores.append(metrics.score)
            if metrics.time:
                times.append(metrics.time)

        if scores:
            self.quality_mean = np.mean(scores)
            self.quality_std = np.mean(scores)
        else:
            self.quality_mean = None
            self.quality_std = None

        if times:
            self.time_mean = np.mean(times)
        else:
            self.time_mean = None

    def __repr__(self):
        return f'PipelineArm[steps={self.steps}]'
