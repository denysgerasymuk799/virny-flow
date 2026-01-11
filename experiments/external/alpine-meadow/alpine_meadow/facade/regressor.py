"""Facade regressor class which conforms to the sklearn interface."""
from alpine_meadow.common import TaskKeyword, PerformanceMetric
from .base import BaseFacade


class AMRegressor(BaseFacade):
    """
    The facade class for regressor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._task_type = TaskKeyword.Value('REGRESSION')
        if 'metric' in kwargs:
            self._metric = PerformanceMetric.Value(kwargs['metric'].upper())
        else:
            self._metric = PerformanceMetric.Value('ROOT_MEAN_SQUARED_ERROR')
