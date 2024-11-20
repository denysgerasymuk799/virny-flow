import uuid
import random
import numpy as np
from munch import DefaultMunch
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace
from virny.configs.constants import *
from virny.custom_classes.metrics_composer import MetricsComposer

from virny_flow.configs.constants import StageName, NO_FAIRNESS_INTERVENTION
from virny_flow.configs.structs import BOAdvisorConfig, Task, LogicalPipeline, PhysicalPipeline
from virny_flow.configs.component_configs import (NULL_IMPUTATION_CONFIG, FAIRNESS_INTERVENTION_CONFIG_SPACE,
                                                  get_models_params_for_tuning)


# Key - metric name from Virny, value - an operation to create a loss for minimization
METRIC_TO_LOSS_ALIGNMENT = {
    # Accuracy metrics
    F1: "reverse",
    ACCURACY: "reverse",
    # Stability metrics
    STD: None,
    IQR: None,
    JITTER: None,
    LABEL_STABILITY: "reverse",
    # Uncertainty metrics
    ALEATORIC_UNCERTAINTY: None,
    EPISTEMIC_UNCERTAINTY: None,
    OVERALL_UNCERTAINTY: None,
    # Error disparity metrics
    EQUALIZED_ODDS_TPR: "abs",
    EQUALIZED_ODDS_TNR: "abs",
    EQUALIZED_ODDS_FPR: "abs",
    EQUALIZED_ODDS_FNR: "abs",
    DISPARATE_IMPACT: "reverse&abs",
    STATISTICAL_PARITY_DIFFERENCE: "abs",
    ACCURACY_DIFFERENCE: "abs",
    # Stability disparity metrics
    LABEL_STABILITY_RATIO: "reverse&abs",
    LABEL_STABILITY_DIFFERENCE: "abs",
    IQR_DIFFERENCE: "abs",
    STD_DIFFERENCE: "abs",
    STD_RATIO: "reverse&abs",
    JITTER_DIFFERENCE: "abs",
    # Uncertainty disparity metrics
    OVERALL_UNCERTAINTY_DIFFERENCE: "abs",
    OVERALL_UNCERTAINTY_RATIO: "reverse&abs",
    EPISTEMIC_UNCERTAINTY_DIFFERENCE: "abs",
    EPISTEMIC_UNCERTAINTY_RATIO: "reverse&abs",
    ALEATORIC_UNCERTAINTY_DIFFERENCE: "abs",
    ALEATORIC_UNCERTAINTY_RATIO: "reverse&abs",
}


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


def select_next_logical_pipeline(logical_pipelines, exploration_factor: float):
    """
    Select the next promising logical pipeline.

    The algorithm is based on the Algorithm 3 (NextLogicalPlan) from the following paper:
     Shang, Zeyuan, et al. "Democratizing data science through interactive curation of ml pipelines."
     Proceedings of the 2019 international conference on management of data. 2019.
    """
    never_explored_lps = [lp for lp in logical_pipelines if lp.score == 0.0]

    # [Exploration] With the likelihood 1 - exploration_factor, randomly select a logical pipeline which we have never run before.
    if random.random() >= exploration_factor and len(never_explored_lps) > 0:
        selected_pipeline = np.random.choice(never_explored_lps)

    # [Exploitation] With likelihood exploration_factor, select a general logical pipeline, which worked well in the past.
    else:
        selected_pipeline = softmax_selection(logical_pipelines)

    return selected_pipeline


def parse_config_space(config_space):
    return {k: ("None" if v is None else v) for k, v in config_space.items()}


def get_suggestion(config_advisor: AsyncBatchAdvisor):
    suggestion = config_advisor.get_suggestion()
    component_params = suggestion.get_dictionary()

    return {k: (None if v == "None" else v) for k, v in component_params.items()}


def get_objective_losses(metrics_dct: dict, objectives: list, model_name: str, sensitive_attributes_dct: dict):
    model_overall_metrics_df = metrics_dct[model_name]

    metrics_composer = MetricsComposer(metrics_dct, sensitive_attributes_dct)
    models_composed_metrics_df = metrics_composer.compose_metrics()
    models_composed_metrics_df = models_composed_metrics_df[models_composed_metrics_df.Model_Name == model_name]

    # OpenBox minimizes the objective
    losses = []
    reversed_objectives = []
    for objective in objectives:
        print("objective:", objective)
        metric, group = objective['metric'], objective['group']
        if group == "overall":
            metric_value = model_overall_metrics_df[model_overall_metrics_df.Metric == metric][group].values[0]
        else:
            metric_value = models_composed_metrics_df[models_composed_metrics_df.Metric == metric][group].values[0]

        # Create a loss to minimize based on the metric
        loss = None
        operation = METRIC_TO_LOSS_ALIGNMENT[metric]
        if operation == "abs":
            loss = abs(metric_value)
        elif operation == "reverse":
            loss = 1 - metric_value
        elif operation == "reverse&abs":
            loss = abs(1 - metric_value)

        losses.append(loss)
        reversed_objectives.append(1 - loss)

    result = dict(objectives=losses, reversed_objectives=reversed_objectives)
    return result


def get_config_advisor(logical_pipeline, bo_advisor_config):
    # Create a config space based on config spaces of each stage
    config_space = ConfigurationSpace()
    for stage in StageName:
        components = logical_pipeline.components
        if stage == StageName.null_imputation:
            stage_config_space = NULL_IMPUTATION_CONFIG[components[stage.value]]['config_space']
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

    return config_advisor, config_space


def select_next_physical_pipelines(logical_pipeline: LogicalPipeline, lp_to_advisor: dict,
                                   bo_advisor_config: BOAdvisorConfig, exp_config: DefaultMunch):
    config_advisor, config_space = (
        (lp_to_advisor[logical_pipeline.logical_pipeline_name]["config_advisor"], lp_to_advisor[logical_pipeline.logical_pipeline_name]["config_space"])
            if lp_to_advisor.get(logical_pipeline.logical_pipeline_name, None) is not None
            else get_config_advisor(logical_pipeline, bo_advisor_config))

    # Create physical pipelines based on MO-BO suggestions
    physical_pipelines = []
    num_new_tasks = min(exp_config.num_pp_candidates, exp_config.max_trials - logical_pipeline.num_trials)
    for idx in range(num_new_tasks):
        suggestion = get_suggestion(config_advisor)
        null_imputer_params = {k.replace('mvi__', '', 1): v for k, v in suggestion.items() if k.startswith('mvi__')}
        fairness_intervention_params = {k.replace('fi__', '', 1): v for k, v in suggestion.items() if k.startswith('fi__')}
        model_params = {k.replace('model__', '', 1): v for k, v in suggestion.items() if k.startswith('model__')}

        physical_pipeline = PhysicalPipeline(physical_pipeline_uuid=str(uuid.uuid4()),
                                             logical_pipeline_uuid=logical_pipeline.logical_pipeline_uuid,
                                             logical_pipeline_name=logical_pipeline.logical_pipeline_name,
                                             exp_config_name=exp_config.exp_config_name,
                                             suggestion=suggestion,
                                             null_imputer_params=null_imputer_params,
                                             fairness_intervention_params=fairness_intervention_params,
                                             model_params=model_params)
        physical_pipelines.append(physical_pipeline)

    new_tasks = [Task(task_uuid=str(uuid.uuid4()),
                      exp_config_name=exp_config.exp_config_name,
                      objectives=exp_config.objectives,
                      physical_pipeline=physical_pipeline)
                 for physical_pipeline in physical_pipelines]
    lp_to_advisor[logical_pipeline.logical_pipeline_name] = {"config_advisor": config_advisor, "config_space": config_space}

    return new_tasks
