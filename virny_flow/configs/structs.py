from typing import List
from dataclasses import dataclass
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
    logical_pipeline_id: int
    logical_pipeline_name: str
    components: dict
    score: float
    risk_factor: float
    pipeline_quality_mean: float
    pipeline_quality_std: float
    pipeline_execution_cost: float
    norm_pipeline_quality_mean: float
    norm_pipeline_quality_std: float
    norm_pipeline_execution_cost: float


@dataclass
class PhysicalPipeline:
    physical_pipeline_id: int
    logical_pipeline_id: int
    logical_pipeline_name: str
    null_imputer_params: dict
    fairness_intervention_params: dict
    model_params: dict
    preprocessing: str = 'cat: OneHotEncoder, num: StandardScaler'


@dataclass
class Task:
    task_id: int
    physical_pipeline: PhysicalPipeline
    task_status: TaskStatus
