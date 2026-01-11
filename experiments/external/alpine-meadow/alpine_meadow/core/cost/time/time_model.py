"""Time model."""

from alpine_meadow.utils.performance import time_calls


class TimeModel:
    """
    The cost model for estimating the execution time of a pipeline arm
    """

    @time_calls
    def estimate_time(self, pipeline_arm):
        """
        Return the estimated running time of the given pipeline arm.
        :param pipeline_arm:
        :return:
        """

        return pipeline_arm.time_mean
