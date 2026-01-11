"""Implementation of the default backend."""

from alpine_meadow.backend.base import BaseBackend
from alpine_meadow.common.config import Config
from .pipeline_executor import DefaultPipelineExecutor


class DefaultBackend(BaseBackend):
    """
    The default backend running inside Alpine Meadow.
    """

    def __init__(self, config: Config):
        self._config = config

    def get_pipeline_executor(self, pipeline, metrics):
        return DefaultPipelineExecutor(pipeline, metrics, backend=self)

    def get_num_workers(self):
        return self._config.evaluation_workers_num

    def get_all_primitives(self):
        from .primitive import LOGICAL_TO_PHYSICAL_TABLE

        return LOGICAL_TO_PHYSICAL_TABLE.keys()
