from dataclasses import dataclass
from virny_flow.configs.constants import TaskStatus


@dataclass
class Task:
    task_id: int
    task_name: str
    task_status: TaskStatus
    stage_id: int


@dataclass
class LogicalPipeline:
    logical_pipeline_id: int
    logical_pipeline: str
    score: float
    pipeline_quality_mean: float
    pipeline_quality_std: float
    pipeline_execution_cost: float
    norm_pipeline_quality_mean: float
    norm_pipeline_quality_std: float
    norm_pipeline_execution_cost: float
