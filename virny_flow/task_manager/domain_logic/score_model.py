import statistics
from openbox.utils.history import Observation

from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import LogicalPipeline
from virny_flow.configs.constants import LOGICAL_PIPELINE_SCORES_TABLE, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE


async def update_logical_pipeline_score_model(exp_config_name: str, objectives_lst: list, observation: Observation,
                                              physical_pipeline_uuid: str, logical_pipeline_uuid: str,
                                              db_client: TaskManagerDBClient):
    # Score model: score = pipeline_quality_mean + risk_factor * pipeline_quality_std / pipeline_execution_cost.

    # Step 1: Get all physical pipeline observations for the defined logical pipeline.
    pp_observations = await db_client.read_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                                 exp_config_name=exp_config_name,
                                                 query={"logical_pipeline_uuid": logical_pipeline_uuid},
                                                 projection={
                                                     "logical_pipeline_uuid": 1,
                                                     "physical_pipeline_uuid": 1,
                                                     "elapsed_time": 1,
                                                     "extra_info": 1,
                                                     "_id": 0,
                                                 })
    # Avoid computation in case the logical pipeline has less than two tested physical pipelines
    # since variance cannot be calculated for one physical pipeline
    if len(pp_observations) < 2:
        return

    # Step 2: Get the logical pipeline from DB.
    logical_pipeline_record = await db_client.read_one_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                             exp_config_name=exp_config_name,
                                                             query={"logical_pipeline_uuid": logical_pipeline_uuid})
    logical_pipeline = LogicalPipeline.from_dict(logical_pipeline_record)

    # Step 3: Compute pipeline_quality_mean, pipeline_quality_std, pipeline_execution_cost for the defined logical pipeline.
    #   For that, calculate mean and std for each objective and average cost based on all physical pipelines
    #   related to the defined logical pipeline.
    # TODO: it can be calculated based on the new observation for this logical pipeline instead of reading all pps for this lp.
    #  Think how to optimise it.
    pipeline_execution_cost = statistics.mean([pp_observation["elapsed_time"] for pp_observation in pp_observations])
    pipeline_quality_mean = dict()
    pipeline_quality_std = dict()
    for idx, objective in enumerate(objectives_lst):
        objective_mean = statistics.mean([pp_observation["extra_info"]["reversed_objectives"][idx] for pp_observation in pp_observations])
        objective_std = statistics.stdev([pp_observation["extra_info"]["reversed_objectives"][idx] for pp_observation in pp_observations])
        pipeline_quality_mean[objective["name"]] = objective_mean
        pipeline_quality_std[objective["name"]] = objective_std + 0.000_001  # To avoid zero division

    logical_pipeline.pipeline_execution_cost = pipeline_execution_cost
    logical_pipeline.pipeline_quality_mean = pipeline_quality_mean
    logical_pipeline.pipeline_quality_std = pipeline_quality_std

    # Step 4: Compute the final score for the defined logical pipeline.
    #   score = pipeline_quality_mean + risk_factor * pipeline_quality_std / pipeline_execution_cost
    final_score = 0.0
    risk_factor = logical_pipeline.risk_factor
    for idx, objective in enumerate(objectives_lst):
        score_per_objective = pipeline_quality_mean[objective["name"]] + risk_factor * pipeline_quality_std[objective["name"]] / pipeline_execution_cost
        final_score += objective['weight'] * score_per_objective

    # Step 5: Compute compound pipeline quality for the current physical pipeline
    compound_pp_quality = 0.0
    for idx, objective in enumerate(objectives_lst):
        compound_pp_quality += objective['weight'] * observation.extra_info['reversed_objectives'][idx]

    await db_client.update_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                 exp_config_name=exp_config_name,
                                 condition={"physical_pipeline_uuid": physical_pipeline_uuid},
                                 update_val_dct={"compound_pp_quality": compound_pp_quality})

    # Step 6: Update the scores in DB
    await db_client.update_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                 exp_config_name=exp_config_name,
                                 condition={"logical_pipeline_uuid": logical_pipeline_uuid},
                                 update_val_dct={"score": final_score,
                                                 "pipeline_quality_mean": pipeline_quality_mean,
                                                 "pipeline_quality_std": pipeline_quality_std,
                                                 "pipeline_execution_cost": pipeline_execution_cost})
