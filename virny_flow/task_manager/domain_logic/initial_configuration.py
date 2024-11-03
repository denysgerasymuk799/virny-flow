import pathlib
from dataclasses import asdict
from datetime import datetime, timezone

from .utils import create_config_obj
from ..database.data_models import Task, LogicalPipeline
from virny_flow.configs.constants import (TaskStatus, STAGE_SEPARATOR, NO_FAIRNESS_INTERVENTION,
                                          TASK_QUEUE_TABLE, LOGICAL_PIPELINE_SCORES_TABLE)


async def create_initial_config_state(db_client):
    # Read an experimental config
    config_yaml_path = pathlib.Path(__file__).parent.parent.joinpath('.', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(config_yaml_path=config_yaml_path)

    # Terminate if the defined exp_config already was executed
    if await db_client.check_record_exists(query={"exp_config_name": exp_config.exp_config_name}):
        print("Experimental config already exists in the database. "
              "Please check the name of your exp_config_name in exp_config.yaml")
        return

    # Create tasks for each stages
    stage1 = []
    stage2 = []
    stage3 = []
    for null_imputer in exp_config.null_imputers:
        stage1.append(null_imputer)
        for fairness_intervention in exp_config.fairness_interventions + [NO_FAIRNESS_INTERVENTION]:
            stage2.append(f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}')
            for model in exp_config.models:
                stage3.append(f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}{STAGE_SEPARATOR}{model}')

    initial_tasks = stage1 + stage2
    logical_pipelines = stage3

    # Save initial tasks in the database
    datetime_now = datetime.now(timezone.utc)
    initial_task_records = [asdict(
                                Task(task_id=idx + 1,
                                     task_name=task_name,
                                     task_status=TaskStatus.READY.value if task_name.count(STAGE_SEPARATOR) == 0 else TaskStatus.BLOCKED.value,
                                     stage_id=task_name.count(STAGE_SEPARATOR) + 1))
                            for idx, task_name in enumerate(initial_tasks)]
    await db_client.write_records_into_db(collection_name=TASK_QUEUE_TABLE,
                                          records=initial_task_records,
                                          static_values_dct={
                                              "exp_config_name": exp_config.exp_config_name,
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "tag": 'OK'
                                          })

    # Save logical pipelines in the database
    logical_pipeline_records = [asdict(
                                    LogicalPipeline(logical_pipeline_id=idx + 1,
                                                    logical_pipeline=logical_pipeline,
                                                    score=None,
                                                    pipeline_quality_mean=None,
                                                    pipeline_quality_std=None,
                                                    pipeline_execution_cost=None,
                                                    norm_pipeline_quality_mean=None,
                                                    norm_pipeline_quality_std=None,
                                                    norm_pipeline_execution_cost=None))
                                for idx, logical_pipeline in enumerate(logical_pipelines)]
    await db_client.write_records_into_db(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                          records=logical_pipeline_records,
                                          static_values_dct={
                                              "exp_config_name": exp_config.exp_config_name,
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "tag": 'OK'
                                          })

    print('The initial state for the exp_config has been successfully created')
