from typing import List
from dataclasses import dataclass, field
from sklearn.impute import SimpleImputer
from openbox.utils.history import History

from virny_flow.configs.constants import TaskStatus


@dataclass
class MixedImputer:
    num_imputer: SimpleImputer
    cat_imputer: SimpleImputer


@dataclass
class BOAdvisorConfig:
    batch_size: int = 4
    batch_strategy: str = 'default'
    num_objectives: int = 1
    num_constraints: int = 0
    sample_strategy: str = 'bo'
    max_trials: int = 100
    max_runtime_per_trial: int = None
    surrogate_type: str = 'auto'
    acq_type: str = 'auto'
    acq_optimizer_type: str = 'auto'
    initial_runs: int = 3
    init_strategy: str = 'random_explore_first'
    initial_configurations=None
    ref_point=None
    transfer_learning_history: List[History] = None
    logging_dir: str = 'logs'
    task_id: str = 'OpenBox'
    random_state: int = None


@dataclass
class LogicalPipeline:
    logical_pipeline_uuid: str
    logical_pipeline_name: str
    components: dict
    risk_factor: float
    num_trials: int
    score: float
    pipeline_quality_mean: float
    pipeline_quality_std: float
    pipeline_execution_cost: float
    norm_pipeline_quality_mean: float
    norm_pipeline_quality_std: float
    norm_pipeline_execution_cost: float


@dataclass
class PhysicalPipeline:
    physical_pipeline_uuid: str
    logical_pipeline_uuid: str
    logical_pipeline_name: str
    null_imputer_params: dict
    fairness_intervention_params: dict
    model_params: dict
    preprocessing: dict = field(default_factory=lambda: {'cat': 'OneHotEncoder', 'num': 'StandardScaler'})


@dataclass
class Task:
    task_uuid: str
    physical_pipeline: PhysicalPipeline
    task_status: TaskStatus
