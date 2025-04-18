import math
import numpy as np
from numba import jit
from openbox.utils.history import Observation

from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import LogicalPipeline
from virny_flow.configs.constants import LOGICAL_PIPELINE_SCORES_TABLE, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE


@jit(nopython=True)
def get_new_mean_based_on_old_mean(new_value, old_mean, n):
    return old_mean + (new_value - old_mean) / (n + 1)


@jit(nopython=True)
def get_new_std_based_on_old_std(new_value, old_std, new_mean, old_mean, n):
    # Based on this derivation:
    # https://math.stackexchange.com/questions/775391/can-i-calculate-the-new-standard-deviation-when-adding-a-value-without-knowing-t
    new_variance = ((n - 1) * (old_std ** 2) + (new_value - new_mean) * (new_value - old_mean)) / n
    return math.sqrt(new_variance)


@jit(nopython=True)
def compute_final_lp_score(pipeline_quality_mean: np.array, pipeline_quality_std: np.array,
                           pipeline_execution_cost: float, objective_weights: np.array, risk_factor: float):
    # Compute the final score for the defined logical pipeline.
    #   score = pipeline_quality_mean + risk_factor * pipeline_quality_std / pipeline_execution_cost
    final_score = 0.0
    for idx in range(len(objective_weights)):
        score_per_objective = pipeline_quality_mean[idx] + risk_factor * pipeline_quality_std[idx] / pipeline_execution_cost
        final_score += objective_weights[idx] * score_per_objective
    return final_score


@jit(nopython=True)
def compute_compound_pp_quality(objective_weights: np.array, reversed_objectives: np.array):
    compound_pp_quality = 0.0
    for idx in range(len(objective_weights)):
        compound_pp_quality += objective_weights[idx] * reversed_objectives[idx]

    return compound_pp_quality


async def update_logical_pipeline_score_model(exp_config_name: str, objectives_lst: list, observation: Observation,
                                              logical_pipeline_uuid: str, db_client: TaskManagerDBClient, run_num: int):
    # Score model: score = pipeline_quality_mean + risk_factor * pipeline_quality_std / pipeline_execution_cost.

    # Step 1: Get the logical pipeline from DB.
    logical_pipeline_record = await db_client.read_one_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                             exp_config_name=exp_config_name,
                                                             run_num=run_num,
                                                             query={"logical_pipeline_uuid": logical_pipeline_uuid})
    logical_pipeline = LogicalPipeline.from_dict(logical_pipeline_record)
    old_pipeline_execution_cost = logical_pipeline.pipeline_execution_cost
    old_pipeline_quality_mean = logical_pipeline.pipeline_quality_mean
    old_pipeline_quality_std = logical_pipeline.pipeline_quality_std

    # Step 2: Compute pipeline_quality_mean, pipeline_quality_std, pipeline_execution_cost for the defined logical pipeline.
    #   For that, calculate mean and std for each objective and average cost based on all physical pipelines
    #   related to the defined logical pipeline.
    if logical_pipeline.num_completed_pps == 0:
        pipeline_execution_cost = observation.elapsed_time
        pipeline_quality_mean = dict()
        pipeline_quality_std = dict()
        for idx, objective in enumerate(objectives_lst):
            objective_mean = observation.extra_info["reversed_objectives"][idx]
            objective_std = 0.0
            pipeline_quality_mean[objective["name"]] = objective_mean
            pipeline_quality_std[objective["name"]] = objective_std + 0.000_001  # To avoid zero division
    else:
        # Rederive new metrics based on old metrics and new value
        pipeline_execution_cost = get_new_mean_based_on_old_mean(new_value=observation.elapsed_time,
                                                                 old_mean=old_pipeline_execution_cost,
                                                                 n=logical_pipeline.num_completed_pps)
        pipeline_quality_mean = dict()
        pipeline_quality_std = dict()
        for idx, objective in enumerate(objectives_lst):
            objective_mean = get_new_mean_based_on_old_mean(new_value=observation.extra_info["reversed_objectives"][idx],
                                                            old_mean=old_pipeline_quality_mean[objective["name"]],
                                                            n=logical_pipeline.num_completed_pps)
            objective_std = get_new_std_based_on_old_std(new_value=observation.extra_info["reversed_objectives"][idx],
                                                         old_std=old_pipeline_quality_std[objective["name"]],
                                                         old_mean=old_pipeline_quality_mean[objective["name"]],
                                                         new_mean=objective_mean,
                                                         n=logical_pipeline.num_completed_pps)
            pipeline_quality_mean[objective["name"]] = objective_mean
            pipeline_quality_std[objective["name"]] = objective_std + 0.000_001  # To avoid zero division

    # Step 3: Compute the final score for the defined logical pipeline
    objective_weights_arr = np.array([obj['weight'] for obj in objectives_lst])
    pipeline_quality_mean_arr = np.array([pipeline_quality_mean[obj["name"]] for obj in objectives_lst])
    pipeline_quality_std_arr = np.array([pipeline_quality_std[obj["name"]] for obj in objectives_lst])
    final_score = compute_final_lp_score(pipeline_quality_mean=pipeline_quality_mean_arr,
                                         pipeline_quality_std=pipeline_quality_std_arr,
                                         pipeline_execution_cost=pipeline_execution_cost,
                                         objective_weights=objective_weights_arr,
                                         risk_factor=logical_pipeline.risk_factor)

    # Step 4: Compute compound pipeline quality for the current physical pipeline
    compound_pp_quality = compute_compound_pp_quality(objective_weights=objective_weights_arr,
                                                      reversed_objectives=np.array(observation.extra_info['reversed_objectives']))

    # Step 5: Update the scores in DB
    await db_client.update_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                 exp_config_name=exp_config_name,
                                 run_num=run_num,
                                 condition={"logical_pipeline_uuid": logical_pipeline_uuid},
                                 update_val_dct={"score": final_score,
                                                 "pipeline_quality_mean": pipeline_quality_mean,
                                                 "pipeline_quality_std": pipeline_quality_std,
                                                 "pipeline_execution_cost": pipeline_execution_cost,
                                                 "num_completed_pps": logical_pipeline.num_completed_pps + 1})
    return compound_pp_quality
