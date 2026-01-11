"""Facade classifier class which conforms to the sklearn interface."""
from alpine_meadow.common import TaskKeyword, PerformanceMetric
from .base import BaseFacade


class AMClassifier(BaseFacade):
    """
    The facade class for classifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._task_type = TaskKeyword.Value('CLASSIFICATION')
        if 'metric' in kwargs:
            self._metric = PerformanceMetric.Value(kwargs['metric'].upper())
        else:
            self._metric = PerformanceMetric.Value('F1_MACRO')
        if 'class_weights' in kwargs:
            self._class_weights = kwargs['class_weights']
