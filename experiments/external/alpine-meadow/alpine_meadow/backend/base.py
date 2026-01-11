"""Alpine Meadow base backend class."""

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

from alpine_meadow.common.metric import is_better_score, score_to_error


# test result stores the outputs (i.e., predictions) besides the metrics
TrainResult = namedtuple('TrainResult', ['metrics', 'outputs'])
TrainResult.__new__.__defaults__ = (None,) * len(TrainResult._fields)


# test result stores the outputs (i.e., predictions) besides the metrics
TestResult = namedtuple('TestResult', ['metrics', 'outputs'])
TestResult.__new__.__defaults__ = (None,) * len(TestResult._fields)


# score result stores the scores (i.e., accuracies) besides the metrics
ScoreResult = namedtuple('ScoreResult', ['metrics', 'scores'])
ScoreResult.__new__.__defaults__ = (None,) * len(ScoreResult._fields)


class BasePipelineExecutor(ABC):
    """
    Base class for executing pipelines, including training, testing (i.e., producing predictions)
    and scoring (i.e., computing accuracy).
    """

    def __init__(self, pipeline, metrics):
        self._pipeline = pipeline
        self._metrics = metrics
        self._validation_method = None
        self._scores = {}

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def metrics(self):
        return self._metrics

    @property
    def validation_method(self):
        return self._validation_method

    @validation_method.setter
    def validation_method(self, validation_method):
        self._validation_method = validation_method

    @property
    def scores(self):
        return self._scores

    def get_score(self, metric=None, to_utility=False, to_error=False):
        """
        Get the score based on the given metric, if there are multiple scores for the same metric,
        we will return the average. If both `to_utility` and `to_error` are False, it returns the
        raw score.
        :param metric: PerformanceMetric (e.g., accuracy)
        :param to_utility: return a utility score (i.e., thg higher the better) if it is True
        :param to_error: return an error score (i.e., the lower the better) if it is True
        :return:
        """

        scores = self.get_scores(metric, to_utility, to_error)
        if scores:
            if isinstance(scores[0], dict):
                aggregated_scores = {}
                for score_dict in scores:
                    for key, value in score_dict.items():
                        if key not in aggregated_scores:
                            aggregated_scores[key] = []
                        aggregated_scores[key].append(value)

                final_scores = {}
                for key, values in aggregated_scores.items():
                    final_scores[key] = np.mean(values)

                return final_scores

            return np.mean(scores)
        return None

    def get_scores(self, metric=None, to_utility=False, to_error=False):
        """
        Get all the scores based on the given metric. If both `to_utility` and `to_error` are False,
        it returns the raw score.
        :param metric: PerformanceMetric (e.g., accuracy)
        :param to_utility: return the utility scores (i.e., thg higher the better) if it is True
        :param to_error: return the error scores (i.e., the lower the better) if it is True
        :return:
        """

        # if metric is None, we use the default metric (the first metric)
        if metric is None:
            metric = self._metrics[0]

        if metric in self._scores:
            scores = self._scores[metric]
            if to_utility:
                new_scores = []
                for score in scores:
                    if is_better_score(metric, 0, 1):
                        new_scores.append(-score)
                    else:
                        new_scores.append(score)
                return new_scores

            if to_error:
                return list(map(lambda score: score_to_error(metric, score), scores))

            return scores
        return []

    def set_scores(self, scores, metric=None):
        """
        Set scores manually.
        :param scores:
        :param metric:
        :return:
        """

        if metric is None:
            metric = self._metrics[0]
        if metric in self._scores:
            self._scores[metric] = scores

    def add_score(self, metric, score):
        """
        Add a new score for the given metric.
        :param metric:
        :param score:
        :return:
        """

        if metric not in self._scores:
            self._scores[metric] = []
        self._scores[metric].append(score)

    def clear_scores(self):
        self._scores = {}

    def merge_scores(self, pipeline_executor):
        for metric, scores in self._scores.items():
            scores.extend(pipeline_executor.get_scores(metric))
        for metric in pipeline_executor.scores:
            if metric not in self._scores:
                self._scores[metric] = pipeline_executor.get_scores(metric)

    def is_better(self, another_pipeline_executor):
        """
        Compare with another pipeline executor, return True if it is better than the other one.
        :param another_pipeline_executor:
        :return:
        """

        metric = self._metrics[0]
        our_score = self.get_score(metric)
        another_score = another_pipeline_executor.get_score(metric)

        # if score is None, then this pipeline shall not be considered
        if another_score is None:
            return True

        if our_score is None:
            return False

        return is_better_score(metric, our_score, another_score)

    @abstractmethod
    def train(self, datasets, **kwargs) -> TrainResult:
        pass

    @abstractmethod
    def test(self, datasets, **kwargs) -> TestResult:
        pass

    def test_proba(self, datasets, **kwargs) -> TestResult:
        return self.test(datasets, **kwargs)

    def score(self, datasets, targets, **kwargs) -> ScoreResult:
        """
        Compute the scores based on the inputs and metrics
        :param datasets:
        :param targets:
        :param kwargs:
        :return:
        """

        from alpine_meadow.common.metric import get_score

        outputs, metrics = self.test(datasets, **kwargs)
        truth = targets[0]
        scores = []
        for metric in self.metrics:
            score = get_score(metric, truth.values, outputs.values, **kwargs)
            self.add_score(metric, score)
            scores.append(score)
        return ScoreResult(scores=scores, metrics=metrics)


class BaseBackend(ABC):
    """
    Base class for the backend executing multiple pipelines.
    """

    @abstractmethod
    def get_pipeline_executor(self, pipeline, metrics):
        pass

    @abstractmethod
    def get_num_workers(self):
        pass

    @abstractmethod
    def get_all_primitives(self):
        pass
