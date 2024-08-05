import pathlib
from datetime import datetime, timezone

from domain_logic.utils import create_config_obj, validate_config
from domain_logic.constants import EXP_PROGRESS_TRACKING_TABLE


def create_execution_plan(db_client):
    # Read an experimental config
    config_yaml_path = pathlib.Path(__file__).parent.parent.joinpath('.', 'configs', 'exp_config.yaml')
    exp_config = create_config_obj(config_yaml_path=config_yaml_path)
    validate_config(exp_config)

    # Create an optimized execution plan
    execution_plan = []
    for null_imputer in exp_config.null_imputers:
        for fairness_intervention in exp_config.fairness_interventions:
            for model in exp_config.models:
                task_name = f'{null_imputer}&{fairness_intervention}&{model}'
                execution_plan.append(task_name)

    # Save the execution plan in the database
    records = [{"task_priority": idx + 1, "task_name": task_name} for idx, task_name in enumerate(execution_plan)]
    db_client.write_records_into_db(collection_name=EXP_PROGRESS_TRACKING_TABLE,
                                    records=records,
                                    static_values_dct={
                                        "config_name": exp_config.config_name,
                                        "done": False,
                                        "create_datetime": datetime.now(timezone.utc),
                                        "update_datetime": datetime.now(timezone.utc),
                                        "tag": 'OK'
                                    })

    print('Execution plan was successfully created')
