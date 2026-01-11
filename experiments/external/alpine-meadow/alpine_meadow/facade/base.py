# pylint: disable=invalid-name
"""Base class for facade classifier and regressor."""
import pandas as pd
import numpy as np

from alpine_meadow.common import Config, Task
from alpine_meadow.core import Optimizer
from alpine_meadow.utils import AMException


class BaseFacade:
    """
    The base class for facade classes, e.g., classifier and regressor
    """

    def __init__(self, **kwargs):
        self._config = Config(**kwargs)
        self._pipeline = None
        self._score = None
        self._task_type = None
        self._metric = None
        self._class_weights = None
        self._task = None

    def fit(self, X, y):
        """
        Given two DataFrame X and y, we fit the facade class
        """

        if len(X) != len(y):
            raise AMException(f"Length of X and y are not matched: {len(X)}, {len(y)}")
        self._task = self._create_task(X, y)
        optimizer = Optimizer(self._task, config=self._config)
        for result in optimizer.optimize():
            self._pipeline = result.pipeline
            self._score = result.score

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = self._convert_to_dataframe(X)
        dataset = self._task.dataset.from_data_frame(X)

        return self._pipeline.test([dataset]).outputs

    @property
    def pipeline_description(self):
        return self._pipeline.pipeline.to_pipeline_desc(human_readable=True)

    @property
    def score(self):
        return self._score

    def _convert_to_dataframe(self, array, is_target=False):
        """
        Convert data to pandas' DataFrame.
        :param array:
        :param is_target:
        :return:
        """

        if isinstance(array, pd.Series):
            return array.to_frame()

        array = np.array(array)
        if len(array.shape) == 1:
            array = array.reshape((-1, 1))
        if len(array.shape) != 2:
            raise AMException(f"array is expected to be 2-dimensional: {array.shape}")

        columns = []
        for column_index in range(array.shape[1]):
            if is_target:
                columns.append(f'target_{column_index}')
            else:
                columns.append(f'feature_{column_index}')

        return pd.DataFrame(data=array, columns=columns)

    def _create_task(self, X, y):
        """
        Create an Alpine Meadow task based on the inputs.
        :param X:
        :param y:
        :return:
        """

        if not isinstance(X, pd.DataFrame):
            X = self._convert_to_dataframe(X)
        if not isinstance(y, pd.DataFrame):
            y = self._convert_to_dataframe(y, is_target=True)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        df = pd.concat([X, y], axis=1, ignore_index=False)
        task = Task([self._task_type], [self._metric], df.columns[-y.shape[1]:],
                    dataset=df, class_weights=self._class_weights)

        return task
