from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace

from virny_flow.configs.constants import StageName, TaskStatus
from virny_flow.configs.structs import BOAdvisorConfig, Task, LogicalPipeline, PhysicalPipeline
from virny_flow.configs.component_configs import (NULL_IMPUTER_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)


def get_suggestion(config_advisor: AsyncBatchAdvisor):
    suggestion = config_advisor.get_suggestion()
    component_params = suggestion.get_dictionary()

    return {k: (None if v == "None" else v) for k, v in component_params.items()}


def get_first_tasks_for_lp(logical_pipeline: LogicalPipeline, lp_to_advisor: dict, bo_advisor_config: BOAdvisorConfig,
                           num_pp_candidates: int):
    # Create a config space based on config spaces of each stage
    config_space = ConfigurationSpace()
    for stage in StageName:
        components = logical_pipeline.components
        if stage == StageName.null_imputation:
            stage_config_space = NULL_IMPUTER_CONFIG[components[stage.value]]['config_space']
        elif stage == StageName.fairness_intervention:
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

    # Create physical pipelines based on MO-BO suggestions
    physical_pipelines = []
    for idx in range(num_pp_candidates):
        suggestion = get_suggestion(config_advisor)
        null_imputer_params = {k.replace('mvi__', ''): v for k, v in suggestion.items() if k.startswith('mvi__')}
        fairness_intervention_params = {k.replace('fi__', ''): v for k, v in suggestion.items() if k.startswith('fi__')}
        model_params = {k.replace('model__', ''): v for k, v in suggestion.items() if k.startswith('model__')}

        physical_pipeline = PhysicalPipeline(physical_pipeline_id=idx + 1,
                                             logical_pipeline_id=logical_pipeline.logical_pipeline_id,
                                             logical_pipeline_name=logical_pipeline.logical_pipeline_name,
                                             null_imputer_params=null_imputer_params,
                                             fairness_intervention_params=fairness_intervention_params,
                                             model_params=model_params)
        physical_pipelines.append(physical_pipeline)

    init_tasks = [Task(task_id=1,
                       task_status=TaskStatus.WAITING.value,
                       physical_pipeline=physical_pipeline)
                  for physical_pipeline in physical_pipelines]
    lp_to_advisor[logical_pipeline.logical_pipeline_name] = {"config_advisor": config_advisor, "trial_num": num_pp_candidates}

    return init_tasks
