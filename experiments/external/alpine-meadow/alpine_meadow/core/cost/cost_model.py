"""Alpine Meadow cost model."""

import numpy as np

from alpine_meadow.utils.performance import time_calls
from .quality import QualityModel
from .time import TimeModel


class CostModel:
    """
    The class of cost model that has two sub cost models: quality and time. It combines
    the returned estimations of both models together to give a score for each pipeline arm.
    """

    def __init__(self, config, state):
        self.config = config
        self.state = state

        self.quality_model = QualityModel()
        self.time_model = TimeModel()

    @time_calls
    def compute_score(self, pipeline_arm):
        """
        Return the score of a pipeline arm where higher is better.
        """

        quality_mean = None
        quality_std = None

        if self.config.enable_learn_from_history:
            external_quality_mean, external_quality_std = self.quality_model.get_external_quality(pipeline_arm)
            if external_quality_mean is not None and external_quality_std is not None:
                quality_mean = self.config.history_weight * external_quality_mean
                quality_std = self.config.history_weight * external_quality_std

        if self.state.mean_score is not None and np.abs(self.state.std_score) > 1e-4:
            current_quality_mean, current_quality_std = self.quality_model.estimate_current_quality(pipeline_arm)
            if current_quality_mean is None or current_quality_std is None:
                current_quality_mean = self.state.mean_score
                current_quality_std = self.state.std_score

            current_quality_mean = (current_quality_mean - self.state.mean_score) / self.state.std_score
            current_quality_std = current_quality_std / self.state.std_score

            if quality_mean is None or quality_std is None:
                quality_mean = current_quality_mean
                quality_std = current_quality_std
            else:
                quality_mean += current_quality_mean
                quality_std += current_quality_std

            # if np.abs(current_quality_mean) > 1e-3:
            #    logger.debug('Global Mean: {}, Std: {}'.format(self.state.mean_score, self.state.std_score))
            #    logger.debug('Current Mean: {}, Std: {}'.format(current_quality_mean, current_quality_std))
            #    logger.debug('History Mean: {}, Std: {}'.format(history_quality_mean, history_quality_std))
            #    logger.debug('Mean: {}, Std: {}'.format(quality_mean, quality_std))

        if quality_mean is None or quality_std is None:
            return np.nan

        if self.config.enable_cost_model:
            execution_time = self.time_model.estimate_time(pipeline_arm)
            if execution_time is None:
                if self.state.mean_time is not None:
                    execution_time = self.state.mean_time
                else:
                    execution_time = 1.0
                # logger.debug('Mean Time: {}'.format(execution_time))
            else:
                # logger.debug('Estimated Time: {}'.format(execution_time))
                pass
            # logger.debug('Time: {}'.format(execution_time))
            quality_std /= execution_time

        score = quality_mean + quality_std * self.config.ucb_delta
        # logger.debug('Mean: {}, Std: {}'.format(quality_mean, quality_std))
        # logger.debug('Score: {}'.format(score))

        return score

    @time_calls
    def compute_scores(self, pipeline_arms):
        """
        Compute the scores of all pipeline arms together
        """

        scores = []
        for pipeline_arm in pipeline_arms:
            score = self.compute_score(pipeline_arm)
            scores.append(score)

        return np.array(scores)

    def show(self, pipeline_arms):
        for pipeline_arm in pipeline_arms:
            history_quality_mean, history_quality_std = self.quality_model.get_history_quality(pipeline_arm)
            current_quality_mean, current_quality_std = self.quality_model.estimate_current_quality(pipeline_arm)
            execution_time = self.time_model.estimate_time(pipeline_arm)

            self.config.logger.debug(
                f'PipelineArm: {pipeline_arm.get_unique_primitives_strs()},'
                f'history_quality: ({history_quality_mean}, {history_quality_std}),'
                f'current_quality: ({current_quality_mean}, {current_quality_std}),'
                f'time: {execution_time}')

    def dumps(self, optimizer):
        """
        Dump the cost model as a string.
        :param optimizer:
        :return:
        """

        cost_model = []
        for pipeline_arm in optimizer.search_space.pipeline_arms:
            history_quality_mean, history_quality_std = self.quality_model.get_external_quality(pipeline_arm)
            current_quality_mean, current_quality_std = self.quality_model.estimate_current_quality(pipeline_arm)
            execution_time = self.time_model.estimate_time(pipeline_arm)

            cost_model.append({
                'history_quality_mean': history_quality_mean,
                'history_quality_std': history_quality_std,
                'current_quality_mean': current_quality_mean,
                'current_quality_std': current_quality_std,
                'execution_time': execution_time,
                'pipeline_arm': pipeline_arm.get_unique_tunable_primitives_strs()
            })

        return cost_model
