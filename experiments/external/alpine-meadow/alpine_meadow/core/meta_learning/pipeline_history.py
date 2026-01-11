"""Alpine Meadow history of pipeline traces."""

import threading

from alpine_meadow.utils.performance import time_calls


class PipelineMetrics:
    """
    The class for the metrics of a pipeline
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._score = None
        self._time = None

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score
        self._pipeline.pipeline_arm.compute_metrics()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time_):
        self._time = time_
        self._pipeline.pipeline_arm.compute_metrics()


class PipelineHistory:
    """
    The history of all pipeline traces.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._next_pipeline_id = 1
        self._pipelines = {}
        self._pipeline_runs = []
        self._pipeline_arms_pipelines = {}

    def save_pipeline(self, pipeline):
        """
        Save a pipeline into the history.
        :param pipeline:
        :return:
        """

        with self._lock:
            pipeline.id = self._next_pipeline_id
            pipeline.metrics = PipelineMetrics(pipeline)
            self._pipelines[self._next_pipeline_id] = pipeline
            self._next_pipeline_id += 1
            if pipeline.pipeline_arm not in self._pipeline_arms_pipelines:
                self._pipeline_arms_pipelines[pipeline.pipeline_arm] = []
            self._pipeline_arms_pipelines[pipeline.pipeline_arm].append(pipeline)

    def get_pipeline(self, pipeline_id):
        with self._lock:
            return self._pipelines.get(pipeline_id, None)

    def get_all_pipelines(self):
        return self._pipelines.values()

    def get_max_pipeline_id(self):
        return self._next_pipeline_id

    def add_pipeline_run(self, pipeline_run):
        with self._lock:
            self._pipeline_runs.append(pipeline_run)

    def get_all_pipeline_runs(self):
        return self._pipeline_runs

    def get_pipelines_by_pipeline_arm(self, pipeline_arm):
        with self._lock:
            return self._pipeline_arms_pipelines.get(pipeline_arm, [])

    @time_calls
    def get_best_k_pipelines_with_scores(self, k):
        """
        Get best k pipelines from the history.
        :param k:
        :return:
        """

        pipelines_with_scores = []
        for pipeline_id in range(1, self.get_max_pipeline_id() + 1):
            pipeline = self.get_pipeline(pipeline_id)
            if pipeline is None:
                continue
            if not pipeline.evaluated:
                continue
            if pipeline.metrics.score:
                pipelines_with_scores.append((pipeline.metrics.score, pipeline))

        from operator import itemgetter

        best_k_pipelines_with_scores = sorted(pipelines_with_scores,
                                              reverse=True, key=itemgetter(0))[:k]

        return best_k_pipelines_with_scores

    @time_calls
    def update_run_history(self, pipeline_id):
        """
        Update the run history object (which is used by Bayesian Optimization) of the given
        pipeline id.
        :param pipeline_id:
        :return:
        """

        from smac.tae import StatusType

        pipeline = self.get_pipeline(pipeline_id)
        run_history = pipeline.pipeline_arm.run_history

        configuration = self.get_pipeline(pipeline_id).configuration
        metrics = pipeline.metrics
        cost = -metrics.score
        time = metrics.time
        run_history.add(config=configuration, cost=cost, time=time, status=StatusType.SUCCESS, seed=0)

    def dump(self):
        """
        Return all the traces as a dict.
        :return:
        """

        with self._lock:
            return {
                'pipelines': list(map(lambda pipeline: pipeline.dumps(),
                                      self.get_all_pipelines())),
                'pipeline_runs': list(map(lambda run: run.dumps(),
                                          self.get_all_pipeline_runs()))
            }

    def __getstate__(self):
        return {
            '_next_pipeline_id': self._next_pipeline_id,
            '_pipelines': self._pipelines
        }

    def __setstate__(self, dict_):
        self._next_pipeline_id = dict_['_next_pipeline_id']
        self._pipelines = dict_['_pipelines']
        self._lock = threading.RLock()
