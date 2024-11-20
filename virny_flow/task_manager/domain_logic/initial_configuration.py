import time
import uuid
from munch import DefaultMunch
from dataclasses import asdict
from datetime import datetime, timezone

from .bayesian_optimization import select_next_logical_pipeline, select_next_physical_pipelines
from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.core.custom_classes.task_queue import TaskQueue
from virny_flow.configs.structs import BOAdvisorConfig, LogicalPipeline
from virny_flow.configs.constants import (StageName, STAGE_SEPARATOR, NO_FAIRNESS_INTERVENTION,
                                          LOGICAL_PIPELINE_SCORES_TABLE, EXP_CONFIG_HISTORY_TABLE)


async def add_new_tasks(exp_config: DefaultMunch, lp_to_advisor: dict, bo_advisor_config: BOAdvisorConfig,
                        db_client: TaskManagerDBClient, task_queue: TaskQueue):
    while True:
        if not await task_queue.has_space_for_next_lp(exp_config_name=exp_config.exp_config_name,
                                                      num_workers=exp_config.num_workers):
            print("Wait until the queue has enough space for next pp candidates...")
            time.sleep(10)
            continue

        # Step 1: Get all logical pipelines, which have num_trials less than max_trials
        query = {"num_trials": {"$lt": exp_config.max_trials}}
        logical_pipeline_records = await db_client.read_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                              exp_config_name=exp_config.exp_config_name,
                                                              query=query)
        logical_pipelines = [LogicalPipeline.from_dict(lp) for lp in logical_pipeline_records]

        if len(logical_pipeline_records) == 0:
            # Terminate in case all work is done
            if await task_queue.is_empty(exp_config_name=exp_config.exp_config_name):
                # Save advisor history for each logical pipeline locally
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                for lp_name, advisor in lp_to_advisor.items():
                    advisor["config_advisor"].save_json(filename=f'history/{exp_config.exp_config_name}/{lp_name}/history_{timestamp}.json')
                return

            # Skip adding new tasks in case all logical pipelines have reached max_trials
            continue

        # Step 2: Get tasks for the selected logical pipeline using MO-BO
        next_logical_pipeline = select_next_logical_pipeline(logical_pipelines=logical_pipelines,
                                                             exploration_factor=exp_config.exploration_factor)
        new_tasks = select_next_physical_pipelines(logical_pipeline=next_logical_pipeline,
                                                   lp_to_advisor=lp_to_advisor,
                                                   bo_advisor_config=bo_advisor_config,
                                                   exp_config=exp_config)

        # Step 3: Update the number of trials for the logical pipeline
        await db_client.increment_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                        exp_config_name=exp_config.exp_config_name,
                                        condition={"logical_pipeline_uuid": next_logical_pipeline.logical_pipeline_uuid},
                                        increment_val_dct={"num_trials": len(new_tasks)})

        # Step 4: Add new tasks to the Task Queue
        for task in new_tasks:
            await task_queue.enqueue(task)


async def create_init_state_for_config(exp_config: DefaultMunch, db_client: TaskManagerDBClient):
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
                        exp_config_name=exp_config.exp_config_name,
                        components=logical_pipeline['components'],
                        risk_factor=exp_config.risk_factor,
                        num_trials=0,
                        score=0.0,
                        best_physical_pipeline_uuid=None,
                        best_compound_pp_improvement=0.0,
                        pipeline_quality_mean={objective['name']: 0.0 for objective in exp_config.objectives},
                        pipeline_quality_std={objective['name']: 0.0 for objective in exp_config.objectives},
                        pipeline_execution_cost=0.0,
                        total_lp_quality_mean_of_means={objective['name']: 0.0 for objective in exp_config.objectives},
                        total_lp_quality_std_of_means={objective['name']: 0.0 for objective in exp_config.objectives},
                        total_lp_quality_mean_of_stds={objective['name']: 0.0 for objective in exp_config.objectives},
                        total_lp_quality_std_of_stds={objective['name']: 0.0 for objective in exp_config.objectives},
                        total_lp_mean_of_execution_costs=0.0,
                        total_lp_std_of_execution_costs=0.0,
                        norm_pipeline_quality_mean={objective['name']: 0.0 for objective in exp_config.objectives},
                        norm_pipeline_quality_std={objective['name']: 0.0 for objective in exp_config.objectives},
                        norm_pipeline_execution_cost=0.0)
        for idx, logical_pipeline in enumerate(logical_pipelines)]
    logical_pipeline_records = [asdict(logical_pipeline_obj) for logical_pipeline_obj in logical_pipeline_objs]
    await db_client.write_records_into_db(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                          records=logical_pipeline_records,
                                          static_values_dct={
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                          })

    # Step 3: Save exp_config in exp_config_history
    await db_client.write_records_into_db(collection_name=EXP_CONFIG_HISTORY_TABLE,
                                          records=[exp_config.toDict()],
                                          static_values_dct={
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                          })

    print(f'The initial state for the {exp_config.exp_config_name} exp_config has been successfully created')
