import statistics

from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import LogicalPipeline
from virny_flow.configs.constants import LOGICAL_PIPELINE_SCORES_TABLE, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE


async def update_logical_pipeline_score_model(exp_config_name: str, objectives_lst: list, logical_pipeline_uuid: str,
                                              db_client: TaskManagerDBClient):
    # Score model: score = pipeline_quality_mean + risk_factor * pipeline_quality_std / pipeline_execution_cost.

    # Step 1: Get all physical pipeline observations for the defined logical pipeline.
    pp_observations = await db_client.read_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                                 exp_config_name=exp_config_name,
                                                 query={"logical_pipeline_uuid": logical_pipeline_uuid})
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
    pipeline_execution_cost = statistics.mean([pp_observation["elapsed_time"] for pp_observation in pp_observations])

    pipeline_quality_mean = dict()
    pipeline_quality_std = dict()
    for idx, objective in enumerate(objectives_lst):
        objective_mean = statistics.mean([pp_observation["extra_info"]["reversed_objectives"][idx] for pp_observation in pp_observations])
        objective_std = statistics.stdev([pp_observation["extra_info"]["reversed_objectives"][idx] for pp_observation in pp_observations])
        pipeline_quality_mean[objective["name"]] = objective_mean
        pipeline_quality_std[objective["name"]] = objective_std

    logical_pipeline.pipeline_execution_cost = pipeline_execution_cost
    logical_pipeline.pipeline_quality_mean = pipeline_quality_mean
    logical_pipeline.pipeline_quality_std = pipeline_quality_std

    # Step 4: Get all other logical pipelines from DB.
    other_logical_pipeline_records = await db_client.read_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                                exp_config_name=exp_config_name,
                                                                query={"logical_pipeline_uuid": {"$ne": logical_pipeline_uuid}})
    other_logical_pipelines = [LogicalPipeline.from_dict(lp) for lp in other_logical_pipeline_records]

    # Step 5: For the defined logical pipeline, compute:
    #         total_lp_quality_mean_of_means, total_lp_quality_std_of_means,
    #         total_lp_quality_mean_of_stds, total_lp_quality_std_of_stds,
    #         total_lp_mean_of_execution_costs, total_lp_std_of_execution_costs.
    #   Example for pipeline_quality_mean: take mean and std of pipeline_quality_mean across all logical pipelines.
    #   Do the same for pipeline_quality_std and pipeline_execution_cost.
    all_pipeline_quality_means = [lp.pipeline_quality_mean for lp in other_logical_pipelines] + [logical_pipeline.pipeline_quality_mean]
    all_pipeline_quality_stds = [lp.pipeline_quality_std for lp in other_logical_pipelines] + [logical_pipeline.pipeline_quality_std]
    all_pipeline_execution_costs = [lp.pipeline_execution_cost for lp in other_logical_pipelines] + [logical_pipeline.pipeline_execution_cost]

    total_lp_mean_of_execution_costs = statistics.mean(all_pipeline_execution_costs)
    total_lp_std_of_execution_costs = statistics.stdev(all_pipeline_execution_costs)

    total_lp_quality_mean_of_means = dict()
    total_lp_quality_std_of_means = dict()
    for idx, objective in enumerate(objectives_lst):
        objective_mean = statistics.mean([pp_observation[objective["name"]] for pp_observation in all_pipeline_quality_means])
        objective_std = statistics.stdev([pp_observation[objective["name"]] for pp_observation in all_pipeline_quality_means])
        total_lp_quality_mean_of_means[objective["name"]] = objective_mean
        total_lp_quality_std_of_means[objective["name"]] = objective_std

    total_lp_quality_mean_of_stds = dict()
    total_lp_quality_std_of_stds = dict()
    for idx, objective in enumerate(objectives_lst):
        objective_mean = statistics.mean([pp_observation[objective["name"]] for pp_observation in all_pipeline_quality_stds])
        objective_std = statistics.stdev([pp_observation[objective["name"]] for pp_observation in all_pipeline_quality_stds])
        total_lp_quality_mean_of_stds[objective["name"]] = objective_mean + 0.000_001 # To avoid zero division
        total_lp_quality_std_of_stds[objective["name"]] = objective_std + 0.000_001  # To avoid zero division

    # Step 6: Compute norm_pipeline_quality_mean, norm_pipeline_quality_std, norm_pipeline_execution_cost for the defined logical pipeline.
    #   Example for norm_pipeline_quality_mean: norm_pipeline_quality_mean = (pipeline_quality_mean  - total_lp_quality_mean_of_means) / total_lp_quality_std_of_means (for each objective).
    #   Do the same for norm_pipeline_quality_std and norm_pipeline_execution_cost.
    norm_pipeline_execution_cost = (pipeline_execution_cost - total_lp_mean_of_execution_costs) / total_lp_std_of_execution_costs

    norm_pipeline_quality_mean = dict()
    norm_pipeline_quality_std = dict()
    for idx, objective in enumerate(objectives_lst):
        norm_objective_mean = (pipeline_quality_mean[objective["name"]] - total_lp_quality_mean_of_means[objective["name"]]) / total_lp_quality_std_of_means[objective["name"]]
        norm_objective_std = (pipeline_quality_std[objective["name"]] - total_lp_quality_mean_of_stds[objective["name"]]) / total_lp_quality_std_of_stds[objective["name"]]
        norm_pipeline_quality_mean[objective["name"]] = norm_objective_mean
        norm_pipeline_quality_std[objective["name"]] = norm_objective_std

    # Step 7: Compute the final score for the defined logical pipeline.
    #   score = norm_pipeline_quality_mean + risk_factor * norm_pipeline_quality_std / norm_pipeline_execution_cost
    score_per_objective = dict()
    risk_factor = logical_pipeline.risk_factor
    for idx, objective in enumerate(objectives_lst):
        score_per_objective[objective["name"]] = norm_pipeline_quality_mean[objective["name"]] + risk_factor * norm_pipeline_quality_std[objective["name"]] / norm_pipeline_execution_cost

    final_score = sum(list(score_per_objective.values()))

    # Step 8: Compute compound pipeline quality for the related physical pipelines
    for pp_observation in pp_observations:
        compound_pp_improvement = 0.0
        norm_compound_pp_improvement = 0.0
        for idx, objective in enumerate(objectives_lst):
            objective_improvement = pp_observation['extra_info']['reversed_objectives'][idx] - pipeline_quality_mean[objective["name"]]
            norm_objective_improvement = (pp_observation['extra_info']['reversed_objectives'][idx] - pipeline_quality_mean[objective["name"]]) / pipeline_quality_std[objective["name"]]
            compound_pp_improvement += objective_improvement
            norm_compound_pp_improvement += norm_objective_improvement

        await db_client.update_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                     exp_config_name=exp_config_name,
                                     condition={"physical_pipeline_uuid": pp_observation["physical_pipeline_uuid"]},
                                     update_val_dct={"compound_pp_improvement": compound_pp_improvement,
                                                     "norm_compound_pp_improvement": norm_compound_pp_improvement})

    # Step 9: Update the scores in DB
    await db_client.update_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                 exp_config_name=exp_config_name,
                                 condition={"logical_pipeline_uuid": logical_pipeline_uuid},
                                 update_val_dct={"score": final_score,
                                                 "pipeline_quality_mean": pipeline_quality_mean,
                                                 "pipeline_quality_std": pipeline_quality_std,
                                                 "pipeline_execution_cost": pipeline_execution_cost,
                                                 "total_lp_quality_mean_of_means": total_lp_quality_mean_of_means,
                                                 "total_lp_quality_std_of_means": total_lp_quality_std_of_means,
                                                 "total_lp_quality_mean_of_stds": total_lp_quality_mean_of_stds,
                                                 "total_lp_quality_std_of_stds": total_lp_quality_std_of_stds,
                                                 "total_lp_mean_of_execution_costs": total_lp_mean_of_execution_costs,
                                                 "total_lp_std_of_execution_costs": total_lp_std_of_execution_costs,
                                                 "norm_pipeline_quality_mean": norm_pipeline_quality_mean,
                                                 "norm_pipeline_quality_std": norm_pipeline_quality_std,
                                                 "norm_pipeline_execution_cost": norm_pipeline_execution_cost})

    await db_client.update_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                 exp_config_name=exp_config_name,
                                 condition={"logical_pipeline_uuid": logical_pipeline_uuid},
                                 update_val_dct={"pipeline_quality_mean": pipeline_quality_mean,
                                                 "pipeline_quality_std": pipeline_quality_std,
                                                 "pipeline_execution_cost": pipeline_execution_cost})
