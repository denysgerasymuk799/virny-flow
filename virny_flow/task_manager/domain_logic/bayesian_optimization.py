import uuid
import random
import numpy as np
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace

from virny_flow.configs.constants import StageName, TaskStatus, NO_FAIRNESS_INTERVENTION
from virny_flow.configs.structs import BOAdvisorConfig, Task, LogicalPipeline, PhysicalPipeline
from virny_flow.configs.component_configs import (NULL_IMPUTER_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)


def compute_metrics(history):
    """
    Compute μ_k (mean performance), δ_k (variance), and c_k (cost) for each logical pipeline k.
    This is a placeholder function and should be implemented based on the specific history format.
    """
    # Placeholder example metrics for demonstration
    metrics = {
        "pipeline_1": {"mu": 0.8, "delta": 0.1, "c": 1},
        "pipeline_2": {"mu": 0.5, "delta": 0.2, "c": 2},
        "pipeline_3": {"mu": 0.6, "delta": 0.3, "c": 1.5},
        # Add more pipelines and their metrics as needed
    }
    return metrics


def softmax_selection(logical_pipelines, temperature=1.0):
    """
    Select a logical pipeline using Softmax (Boltzmann) selection: the higher the score,
     the higher the chance that the general pipeline gets selected.
    """
    scores = [lp.score for lp in logical_pipelines]

    # Apply softmax to compute selection probabilities
    scaled_scores = np.exp(np.array(scores) / temperature)
    probabilities = scaled_scores / np.sum(scaled_scores)

    # Select a pipeline based on computed probabilities
    selected_pipeline = np.random.choice(logical_pipelines, p=probabilities)
    return selected_pipeline


def select_next_logical_pipeline(logical_pipelines, exploration_factor: float, max_trials: int):
    """
    Select the next promising logical pipeline.

    The algorithm is based on the Algorithm 3 (NextLogicalPlan) from the following paper:
     Shang, Zeyuan, et al. "Democratizing data science through interactive curation of ml pipelines."
     Proceedings of the 2019 international conference on management of data. 2019.
    """
    filtered_logical_pipelines = [lp for lp in logical_pipelines if lp.num_trials < max_trials]

    # [Exploitation] With likelihood exploration_factor, select a general logical pipeline, which worked well in the past.
    if random.random() < exploration_factor:
        selected_pipeline = softmax_selection(filtered_logical_pipelines)

    # [Exploration] With the likelihood 1 - exploration_factor, randomly select a logical pipeline which we have never run before.
    else:
        never_explored_lps = [lp for lp in filtered_logical_pipelines if lp.score == 0.0]
        selected_pipeline = np.random.choice(never_explored_lps)

    return selected_pipeline


def get_suggestion(config_advisor: AsyncBatchAdvisor):
    suggestion = config_advisor.get_suggestion()
    component_params = suggestion.get_dictionary()

    return {k: (None if v == "None" else v) for k, v in component_params.items()}


def get_config_advisor(logical_pipeline, bo_advisor_config):
    # Create a config space based on config spaces of each stage
    config_space = ConfigurationSpace()
    for stage in StageName:
        components = logical_pipeline.components
        if stage == StageName.null_imputation:
            stage_config_space = NULL_IMPUTER_CONFIG[components[stage.value]]['config_space']
        elif stage == StageName.fairness_intervention:
            # NO_FAIRNESS_INTERVENTION does not have config space
            if components[stage.value] == NO_FAIRNESS_INTERVENTION:
                continue

            stage_config_space = FAIRNESS_INTERVENTION_CONFIG_SPACE[components[stage.value]]
        else:
            # We set seed to None since we need only a config space
            stage_config_space = get_models_params_for_tuning(models_tuning_seed=None)[components[stage.value]]['config_space']

        config_space.add_hyperparameters(list(stage_config_space.values()))

    _logger_kwargs = {'force_init': False}  # do not init logger in advisor
    config_advisor = AsyncBatchAdvisor(config_space,
                                       num_objectives = bo_advisor_config.num_objectives,
                                       num_constraints = bo_advisor_config.num_constraints,
                                       batch_size = bo_advisor_config.batch_size,
                                       batch_strategy = bo_advisor_config.batch_strategy,
                                       initial_trials = bo_advisor_config.initial_runs,
                                       initial_configurations = bo_advisor_config.initial_configurations,
                                       init_strategy = bo_advisor_config.init_strategy,
                                       transfer_learning_history = bo_advisor_config.transfer_learning_history,
                                       optimization_strategy = bo_advisor_config.sample_strategy,
                                       surrogate_type = bo_advisor_config.surrogate_type,
                                       acq_type = bo_advisor_config.acq_type,
                                       acq_optimizer_type = bo_advisor_config.acq_optimizer_type,
                                       ref_point = bo_advisor_config.ref_point,
                                       task_id = bo_advisor_config.task_id,
                                       output_dir = bo_advisor_config.logging_dir,
                                       random_state = bo_advisor_config.random_state,
                                       logger_kwargs=_logger_kwargs)

    return config_advisor


def select_next_physical_pipelines(logical_pipeline: LogicalPipeline, lp_to_advisor: dict,
                                   bo_advisor_config: BOAdvisorConfig, num_pp_candidates: int):
    config_advisor = lp_to_advisor[logical_pipeline.logical_pipeline_name] \
        if lp_to_advisor.get(logical_pipeline.logical_pipeline_name, None) is not None \
        else get_config_advisor(logical_pipeline, bo_advisor_config)

    # Create physical pipelines based on MO-BO suggestions
    physical_pipelines = []
    for idx in range(num_pp_candidates):
        suggestion = get_suggestion(config_advisor)
        null_imputer_params = {k.replace('mvi__', '', 1): v for k, v in suggestion.items() if k.startswith('mvi__')}
        fairness_intervention_params = {k.replace('fi__', '', 1): v for k, v in suggestion.items() if k.startswith('fi__')}
        model_params = {k.replace('model__', '', 1): v for k, v in suggestion.items() if k.startswith('model__')}

        physical_pipeline = PhysicalPipeline(physical_pipeline_uuid=str(uuid.uuid4()),
                                             logical_pipeline_uuid=logical_pipeline.logical_pipeline_uuid,
                                             logical_pipeline_name=logical_pipeline.logical_pipeline_name,
                                             null_imputer_params=null_imputer_params,
                                             fairness_intervention_params=fairness_intervention_params,
                                             model_params=model_params)
        physical_pipelines.append(physical_pipeline)

    new_tasks = [Task(task_uuid=str(uuid.uuid4()),
                      task_status=TaskStatus.WAITING.value,
                      physical_pipeline=physical_pipeline)
                 for physical_pipeline in physical_pipelines]
    lp_to_advisor[logical_pipeline.logical_pipeline_name] = config_advisor

    return new_tasks
