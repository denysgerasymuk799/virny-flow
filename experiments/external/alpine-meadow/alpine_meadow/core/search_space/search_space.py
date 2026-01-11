"""Alpine Meadow search space."""
import numpy as np
import random

from alpine_meadow.utils.performance import time_calls
from alpine_meadow.utils import AMException
from .pipeline_arm import PipelineArm


class SearchSpace:
    """
    The search space class for all the pipeline arms
    """

    def __init__(self, task, rule_executor, pipeline_history):
        self.task = task
        self.rule_executor = rule_executor
        self.config = self.rule_executor.config

        self._pipeline_arms = [PipelineArm(pipeline_history)]
        self._unique_primitives = set()
        self._num_searched_pipelines = 0

    def _get_pipeline_arm(self, cost_model, deterministic):
        """
        Get a pipeline arm from the search space, deterministically or randomly.
        """

        tags = {}
        logger = self.config.logger

        if not deterministic:
            prob = random.random()
        else:
            prob = 1.0

        # if
        # (1) prob < random_random_threshold: we will pick a pipeline arm randomly
        # (2) random_random_threshold < prob < random_cost_model_threshold: we will pick a pipeline with
        # weighted random based on their scores
        # (3) prob > random_cost_model_threshold: we will pick the pipeline with the maximum score

        if prob < self.config.random_random_threshold:
            if self.config.enable_mutation_rules and random.random() < self.config.mutation_rules_threshold:
                pipeline_arm = self.rule_executor.mutate(self)
                tags['pipeline_arm'] = 'random-data-specific'
            else:
                pipeline_arm = np.random.choice(self._pipeline_arms)
                tags['pipeline_arm'] = 'random-general'

            logger.debug(f'Picking pipeline arm with random: {pipeline_arm.get_unique_primitives_strs()}')
            return pipeline_arm, tags

        scores = cost_model.compute_scores(self._pipeline_arms)
        # logger.debug('Scores: {}'.format(scores))

        if prob < self.config.random_cost_model_threshold:
            score_threshold_candidates_num = min(self.config.score_threshold_candidates_num, len(scores))
            scores[np.isnan(scores)] = 0.0
            scores[np.isinf(scores)] = 0.0
            score_threshold = max(
                0.0,
                np.percentile(scores, 100 - float(score_threshold_candidates_num * 100 / len(scores))))
            # logger.debug('Score threshold: {}, {}'.format(score_threshold, (scores >= score_threshold).sum()))
            scores[scores < score_threshold] = 0.0
            # logger.debug('Scores: {}'.format(scores))
            # logger.debug('Sum: {}, {}'.format(scores.sum(), scores.sum() > 1e-4))

            if scores.sum() > 1e-4:
                scores /= scores.sum()
                # logger.debug('Prob: {}'.format(scores))
                pipeline_arm = np.random.choice(self._pipeline_arms, p=scores)
                tags['pipeline_arm'] = 'cost_model-weighed_random'
            else:
                pipeline_arm = np.random.choice(self._pipeline_arms)
                tags['pipeline_arm'] = 'cost_model-random'

            logger.debug(f'Picking pipeline arm with weight-random '
                         f'in cost model: {pipeline_arm.get_unique_primitives_strs()}')
            return pipeline_arm, tags

        # deterministic: pick the pipeline with maximum score
        pipeline_arm = self._pipeline_arms[np.argmax(scores)]
        tags['pipeline_arm'] = 'cost_model-maximum'
        logger.debug(f'Picking pipeline arm with max-score '
                     f'in cost model: {pipeline_arm.get_unique_primitives_strs()}')
        return pipeline_arm, tags

    @time_calls
    def get_k_pipeline_arms(self, cost_model, k):
        """
        Get top k pipeline arms.
        :param cost_model:
        :param k:
        :return: top k pipeline arms
        """

        pipeline_arms = []
        for _ in range(k):
            pipeline_arm, tags = self._get_pipeline_arm(cost_model, self.config.enable_deterministic)
            pipeline_arms.append((pipeline_arm, tags))

        return pipeline_arms

    def add_pipeline_arm(self, pipeline_arm: PipelineArm):
        self._pipeline_arms.append(pipeline_arm)
        for primitive in pipeline_arm.get_unique_primitives_strs():
            self._unique_primitives.add(primitive)

    @property
    def pipeline_arms(self):
        return self._pipeline_arms

    @property
    def unique_primitives(self):
        return self._unique_primitives

    @property
    def num_searched_pipelines(self):
        return self._num_searched_pipelines

    def increase_num_searched_pipelines(self, count):
        self._num_searched_pipelines += count

    def enforce(self, enforcement_rule):
        """
        Apply the enforcement rule in the search space.
        """

        def enforce_pipeline_arm(pipeline_arm):
            if pipeline_arm.complete:
                return True

            return enforcement_rule.enforce(self.task, pipeline_arm)

        num_pipeline_arms = len(self._pipeline_arms)
        self._pipeline_arms = list(filter(enforce_pipeline_arm, self._pipeline_arms))
        if not self._pipeline_arms:
            raise AMException(enforcement_rule.__name__ + " cannot pass!")
        self.config.logger.debug(f'Enforcement rules picked {len(self._pipeline_arms)}'
                                 f' of {num_pipeline_arms} pipeline arms')

    def filter_by_primitives(self, including_primitives, excluding_primitives):
        """
        Filter the pipeline arms so that the pipeline arms have to have "including_primitives" and
        have not to have "excluding_primitives".
        :param including_primitives:
        :param excluding_primitives:
        :return:
        """

        if including_primitives:
            including_primitives = set(including_primitives)
            applicable_primitives = []
            for primitive in including_primitives:
                applicable = False
                for pipeline_arm in self._pipeline_arms:
                    pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
                    if primitive in pipeline_arm_primitives:
                        applicable = True
                        break

                if applicable:
                    applicable_primitives.append(primitive)

            def have_primitives(pipeline_arm):
                pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
                return set(applicable_primitives) <= pipeline_arm_primitives

            self._pipeline_arms = list(filter(have_primitives, self._pipeline_arms))

        if excluding_primitives:
            excluding_primitives = set(excluding_primitives)

            def not_have_primitives(pipeline_arm):
                pipeline_arm_primitives = set(pipeline_arm.get_unique_primitives())
                if excluding_primitives.intersection(pipeline_arm_primitives):
                    return False
                return True

            self._pipeline_arms = list(filter(not_have_primitives, self._pipeline_arms))

    def show(self):
        for pipeline_arm in self._pipeline_arms:
            print(pipeline_arm)
            print('*' * 30)
