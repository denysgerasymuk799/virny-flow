"""Alpine Meadow optimizer base classes."""

import time

import numpy as np

from alpine_meadow.utils import AMException


class OptimizerState:
    """
    Optimizer statistics.
    """

    def __init__(self, optimizer):
        import threading

        self._optimizer = optimizer

        # score
        self.mean_score = None
        self.std_score = None
        self.best_score = None
        self.scores = []

        # time
        self.mean_time = None
        self.times = []

        # status
        self.start_time = time.perf_counter()
        self.find_first_pipeline = False
        self.done = False

        # budget
        self.time_limit = None
        self.pipelines_num_limit = None

        # lock
        self._lock = threading.RLock()

    def add_score(self, score):
        with self._lock:
            self.scores.append(score)
            self.mean_score = np.mean(self.scores)
            self.std_score = np.std(self.scores)
            self.best_score = np.max(self.scores)

    def add_time(self, time_):
        with self._lock:
            self.times.append(time_)
            self.mean_time = np.mean(self.times)

    def start(self):
        self.start_time = time.perf_counter()
        self.done = False

    def get_elapsed_time(self):
        return time.perf_counter() - self.start_time

    def get_remaining_time(self):
        if not self.find_first_pipeline:
            return None

        if self.time_limit is not None:
            return self.time_limit - self.get_elapsed_time()
        return None

    def progress(self):
        elapsed_time = self.get_elapsed_time()
        if self.time_limit is not None:
            progress = min(elapsed_time * 100 / self.time_limit, 99.9)
        else:
            if self.pipelines_num_limit is None:
                raise AMException("No limit for time and pipeline nums!")
            progress = min(self._optimizer.metrics['validated_pipelines_num'] * 100 / self.pipelines_num_limit, 99.9)
        return progress


class OptimizerResult:
    """
    Result returned by the `optimize` method of a Optimizer class. It includes the pipeline (executor),
    score (raw), progress (between 0 and 1) and elapsed time since the start.
    """

    def __init__(self, pipeline, score, progress, elapsed_time):
        self.pipeline = pipeline
        self.score = score
        self.progress = progress
        self.elapsed_time = elapsed_time
