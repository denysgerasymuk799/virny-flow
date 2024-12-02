from typing import List
from dataclasses import dataclass, field, fields
from sklearn.impute import SimpleImputer
from openbox.utils.history import History


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
    exp_config_name: str
    components: dict
    risk_factor: float
    num_trials: int
    max_trials: int
    score: float
    best_physical_pipeline_uuid: str
    best_compound_pp_quality: float
    best_compound_pp_improvement: float
    pipeline_quality_mean: dict
    pipeline_quality_std: dict
    pipeline_execution_cost: float
    num_completed_pps: int
    run_num: int
    random_state: int

    @classmethod
    def from_dict(cls, data: dict):
        # Get the set of field names defined in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        # Filter the input dictionary to only include keys that are valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        # Return an instance of the dataclass initialized with the filtered data
        return cls(**filtered_data)


@dataclass
class PhysicalPipeline:
    physical_pipeline_uuid: str
    logical_pipeline_uuid: str
    logical_pipeline_name: str
    exp_config_name: str
    suggestion: dict
    null_imputer_params: dict
    fairness_intervention_params: dict
    model_params: dict
    run_num: int
    random_state: int
    preprocessing: dict = field(default_factory=lambda: {'cat': 'OneHotEncoder', 'num': 'StandardScaler'})

    @classmethod
    def from_dict(cls, data: dict):
        # Get the set of field names defined in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        # Filter the input dictionary to only include keys that are valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        # Return an instance of the dataclass initialized with the filtered data
        return cls(**filtered_data)


@dataclass
class Task:
    task_uuid: str
    exp_config_name: str
    objectives: list
    pipeline_quality_mean: dict
    physical_pipeline: PhysicalPipeline
    run_num: int
    random_state: int

    @classmethod
    def from_dict(cls, data: dict):
        # Get the set of field names defined in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        # Filter the input dictionary to only include keys that are valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        filtered_data["physical_pipeline"] = PhysicalPipeline.from_dict(filtered_data["physical_pipeline"])

        # Return an instance of the dataclass initialized with the filtered data
        return cls(**filtered_data)
