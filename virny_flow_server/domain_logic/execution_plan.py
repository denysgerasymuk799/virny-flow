import pathlib
from datetime import datetime, timezone

from domain_logic.utils import create_config_obj
from domain_logic.constants import EXP_PROGRESS_TRACKING_TABLE, TaskStatus, STAGE_SEPARATOR, NO_FAIRNESS_INTERVENTION


async def create_execution_plan(db_client):
    # Read an experimental config
    config_yaml_path = pathlib.Path(__file__).parent.parent.joinpath('.', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(config_yaml_path=config_yaml_path)

    # Create an optimized execution plan
    stage1 = []
    stage2 = []
    stage3 = []
    for null_imputer in exp_config.null_imputers:
        stage1.append(null_imputer)
        for fairness_intervention in exp_config.fairness_interventions + [NO_FAIRNESS_INTERVENTION]:
            stage2.append(f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}')
            for model in exp_config.models:
                stage3.append(f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}{STAGE_SEPARATOR}{model}')

    execution_plan = stage1 + stage2 + stage3

    # Save the execution plan in the database
    if await db_client.check_record_exists(query={"exp_config_name": exp_config.exp_config_name}):
        print("Experimental config already exists in the database. "
              "Please check the name of your exp_config_name in exp_config.yaml")
        return

    records = [{"task_id": idx + 1,
                "task_name": task_name,
                "task_status": TaskStatus.READY.value if task_name.count(STAGE_SEPARATOR) == 0 else TaskStatus.BLOCKED.value,
                "stage_id": task_name.count(STAGE_SEPARATOR) + 1}
               for idx, task_name in enumerate(execution_plan)]
    datetime_now = datetime.now(timezone.utc)
    await db_client.write_records_into_db(collection_name=EXP_PROGRESS_TRACKING_TABLE,
                                          records=records,
                                          static_values_dct={
                                              "exp_config_name": exp_config.exp_config_name,
                                              "create_datetime": datetime_now,
                                              "update_datetime": datetime_now,
                                              "tag": 'OK'
                                          })

    print('Execution plan was successfully created')
