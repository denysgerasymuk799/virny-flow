"""Alpine Meadow API context."""

from threading import Lock


class APIContext:
    """
    The API context is responsible for the management of metadata.
    """

    def __init__(self):
        import alpine_meadow

        self._lock = Lock()
        self._version = alpine_meadow.__version__
        self._tasks = []
        self._pipelines = {}

    @property
    def version(self):
        return self._version

    def get_tasks(self):
        return self._tasks

    def register_task(self, task):
        with self._lock:
            self._tasks.append(task)

    def get_pipelines(self, task_id):
        with self._lock:
            return self._pipelines[task_id]

    def register_pipelines(self, task_id, pipelines):
        with self._lock:
            if task_id not in self._pipelines:
                self._pipelines[task_id] = []
            self._pipelines[task_id].extend(pipelines)
