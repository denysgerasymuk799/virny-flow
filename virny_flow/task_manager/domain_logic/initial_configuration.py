import uuid
from munch import DefaultMunch
from dataclasses import asdict
from datetime import datetime, timezone

from .bayesian_optimization import select_next_logical_pipeline, select_next_physical_pipelines
from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import BOAdvisorConfig, LogicalPipeline
from virny_flow.configs.constants import (StageName, STAGE_SEPARATOR, NO_FAIRNESS_INTERVENTION,
                                          TASK_QUEUE_TABLE, LOGICAL_PIPELINE_SCORES_TABLE, EXP_CONFIG_HISTORY_TABLE)


async def create_init_state_for_config(exp_config: DefaultMunch, lp_to_advisor: dict,
                                       bo_advisor_config: BOAdvisorConfig, db_client: TaskManagerDBClient):
    # Terminate if the defined exp_config already was executed
    if await db_client.check_record_exists(query={"exp_config_name": exp_config.exp_config_name}):
        print("Experimental config already exists in the database. "
              "Please check the name of your exp_config_name in exp_config.yaml")
        return

    # Step 1: Create all combinations of components to define a list of logical pipelines
    logical_pipelines = []
    for null_imputer in exp_config.null_imputers:
        for fairness_intervention in exp_config.fairness_interventions + [NO_FAIRNESS_INTERVENTION]:
            for model in exp_config.models:
                logical_pipeline = {
                    'name': f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}{STAGE_SEPARATOR}{model}',
                    'components': {
                        StageName.null_imputation.value: null_imputer,
                        StageName.fairness_intervention.value: fairness_intervention,
                        StageName.model_evaluation.value: model,
                    }
                }
                logical_pipelines.append(logical_pipeline)

    # Step 2: Init all logical pipelines with default values in the database
    datetime_now = datetime.now(timezone.utc)
    logical_pipeline_objs = [
        LogicalPipeline(logical_pipeline_uuid=str(uuid.uuid4()),
                        logical_pipeline_name=logical_pipeline['name'],
                        components=logical_pipeline['components'],
                        risk_factor=exp_config.risk_factor,
                        num_trials=0,
                        score=0.0,
                        pipeline_quality_mean=0.0,
                        pipeline_quality_std=0.0,
                        pipeline_execution_cost=0.0,
                        norm_pipeline_quality_mean=0.0,
                        norm_pipeline_quality_std=0.0,
                        norm_pipeline_execution_cost=0.0)
        for idx, logical_pipeline in enumerate(logical_pipelines)]
    logical_pipeline_records = [asdict(logical_pipeline_obj) for logical_pipeline_obj in logical_pipeline_objs]
    await db_client.write_records_into_db(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                          records=logical_pipeline_records,
                                          static_values_dct={
                                              "exp_config_name": exp_config.exp_config_name,
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "deletion_flag": False
                                          })

    # Step 3: Get tasks for the first logical pipeline using MO-BO
    next_logical_pipeline = select_next_logical_pipeline(logical_pipelines=logical_pipeline_objs,
                                                         exploration_factor=exp_config.exploration_factor,
                                                         max_trials=exp_config.max_trials)
    init_tasks = select_next_physical_pipelines(logical_pipeline=next_logical_pipeline,
                                                lp_to_advisor=lp_to_advisor,
                                                bo_advisor_config=bo_advisor_config,
                                                num_pp_candidates=exp_config.num_pp_candidates)
    # Update the number of trials for the selected logical pipeline
    await db_client.increment_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                    condition={"exp_config_name": exp_config.exp_config_name,
                                               "logical_pipeline_uuid": next_logical_pipeline.logical_pipeline_uuid},
                                    increment_val_dct={"num_trials": exp_config.num_pp_candidates})

    # Step 4: Add the initial tasks to the Task Queue
    init_task_records = [asdict(task) for task in init_tasks]
    await db_client.write_records_into_db(collection_name=TASK_QUEUE_TABLE,
                                          records=init_task_records,
                                          static_values_dct={
                                              "exp_config_name": exp_config.exp_config_name,
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "deletion_flag": False
                                          })

    # Step 5: Save exp_config in exp_config_history
    await db_client.write_records_into_db(collection_name=EXP_CONFIG_HISTORY_TABLE,
                                          records=[exp_config.toDict()],
                                          static_values_dct={
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "deletion_flag": False
                                          })

    print(f'The initial state for the {exp_config.exp_config_name} exp_config has been successfully created')
