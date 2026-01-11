"""Rule executor for running the rules to create the search space."""
import time

import numpy as np

from alpine_meadow.core.search_space import SearchSpace
from alpine_meadow.utils import AMException
from alpine_meadow.utils.performance import time_calls
from .primitive import rule as primitive_rule
from .parameter import rule as parameter_rule
from .enforcement import rule as enforcement_rule
from .mutation import rule as mutation_rules


class RuleExecutor:
    """
    The executor class for executing all the rules
    """

    def __init__(self, config):
        self.config = config

        # register primitive rules
        self._primitive_rules = []
        primitive_rule.register_rules(self)

        # register parameter rules
        self._parameter_rules = []
        parameter_rule.register_rules(self)

        # register enforcement rules
        self._enforcement_rules = []
        enforcement_rule.register_rules(self)

        # register tweak rules
        self._mutation_rules = []
        mutation_rules.register_rules(self)

    def register_primitive_rule(self, rule):
        self._primitive_rules.append(rule)

    def register_parameter_rule(self, rule):
        self._parameter_rules.append(rule)

    def register_enforcement_rule(self, rule):
        self._enforcement_rules.append(rule)

    def register_mutation_rule(self, rule):
        self._mutation_rules.append(rule)

    @time_calls
    def execute(self, task, pipeline_history):
        """
        Execute the rules based on the task and pipeline history.
        :param task:
        :param pipeline_history:
        :return:
        """

        logger = self.config.logger

        # compute meta features
        start = time.perf_counter()
        try:
            task.compute_meta_features()
        except BaseException:  # pylint: disable=broad-except
            logger.error(msg='', exc_info=True)
        logger.debug(f'Calculating meta features time: {time.perf_counter() - start}')

        # check the meta features to see if the task makes sense
        meta_features = task.meta_features
        if meta_features is not None and 'NumberOfClasses' in meta_features.keys()\
                and meta_features['NumberOfClasses'].value > 500:
            raise AMException(f"The cardinality of the target column is more than 500: "
                              f"{meta_features['NumberOfClasses'].value}")

        # create search space
        search_space = SearchSpace(task, self, pipeline_history)

        # apply primitive rules
        start = time.perf_counter()
        for rule in self._primitive_rules:
            rule_start = time.perf_counter()
            if rule.predicate(task):
                rule.apply(search_space)
            logger.debug(f'[Primitive Rule] {type(rule)} took {time.perf_counter() - rule_start} seconds')
        logger.debug(f'Applying primitive rules time: {time.perf_counter() - start}')
        search_space.filter_by_primitives(including_primitives=self.config.including_primitives,
                                          excluding_primitives=self.config.excluding_primitives)

        # apply parameter rules
        start = time.perf_counter()
        for rule in self._parameter_rules:
            rule_start = time.perf_counter()
            for pipeline_arm in search_space.pipeline_arms:
                for step in pipeline_arm.steps:
                    if rule.predicate(task, step):
                        rule.apply(task, step)
                        pipeline_arm.parameter_rules.add(rule)
            logger.debug(f'[Parameter Rule] {type(rule)} took {time.perf_counter() - rule_start} seconds')
        logger.debug(f'Applying parameter rules time: {time.perf_counter() - start}')

        # apply enforcement rules
        start = time.perf_counter()
        for rule in self._enforcement_rules:
            rule_start = time.perf_counter()
            if rule.predicate(task):
                search_space.enforce(rule)
            logger.debug(f'[Enforcement Rule] {rule} took {time.perf_counter() - rule_start} seconds')
        logger.debug(f'Applying enforcement rules time: {time.perf_counter() - start}')

        # creating configuration space
        start = time.perf_counter()
        pipeline_arms = search_space.pipeline_arms
        configuration_spaces = {}
        for pipeline_arm in pipeline_arms:
            key = frozenset(pipeline_arm.get_unique_tunable_primitives_strs())
            if key in configuration_spaces:
                pipeline_arm.configuration_space = configuration_spaces[key]
            else:
                pipeline_arm.create_configuration_space()
                configuration_spaces[key] = pipeline_arm.configuration_space
            pipeline_arm.search_space = search_space
            # logger.info('{}, {}, {}'.format(pipeline_arm.get_unique_primitives_strs()[-1],
            #                                 pipeline_arm.external_quality_mean, pipeline_arm.external_quality_std))
        logger.debug(f'Validating pipeline arms time: {time.perf_counter() - start}')

        # search_space.show()
        logger.debug(f'Total pipeline arms num: {len(search_space.pipeline_arms)}')

        return search_space

    def mutate(self, search_space):
        rule = np.random.choice(self._mutation_rules)
        pipeline_arm = rule.apply(search_space)

        return pipeline_arm
