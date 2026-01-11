"""Alpine Meadow runtime information about pipelines, including pipeline description,
pipeline run and primitive run."""


class StepRun:
    """
    Information for running a step, including time (fit/produce) and primitive.
    """

    def __init__(self, **kwargs):
        self.index = None
        self.primitive = None
        self.fit_time = None
        self.produce_time = None

        # initialize
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def dumps(self):
        return {
            'index': self.index,
            'primitive:': self.primitive,
            'fit_time': self.fit_time,
            'produce_time': self.produce_time
        }


class PipelineRun:
    """
    Information for running a pipeline, including start/end time, trained/tested splits,
    and information for running each step.
    """

    def __init__(self, **kwargs):
        self.pipeline = None
        self.context = None
        self.trained_splits = None
        self.test_split = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.tags = []
        self.step_runs = []

        # initialize
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def add_step_run(self, step_run):
        self.step_runs.append(step_run)

    def dumps(self):
        return {
            'pipeline_id': self.pipeline.id,
            'context': self.context,
            'trained_splits': self.trained_splits,
            'test_split': self.test_split,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'tags': self.tags,
            'step_runs': list(map(lambda run: run.dumps(), self.step_runs)),
        }
