import os
import time
import warnings

from virny_flow.configs.constants import FINISH_EXECUTION, NO_READY_TASK
from virny_flow.core.custom_classes.pipeline_evaluator import PipelineEvaluator
from virny_flow.core.custom_classes.worker import Worker


def worker_interface(exp_config, virny_flow_address: str, dataset_config: dict,
                     fairness_intervention_config: dict, models_config: dict):
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Init objects
    pipeline_evaluator = PipelineEvaluator(exp_config=exp_config,
                                           dataset_config=dataset_config,
                                           fairness_intervention_config=fairness_intervention_config,
                                           models_config=models_config)

    # Get an initial task
    virny_flow_client = Worker(address=virny_flow_address)
    task = virny_flow_client.get_task(exp_config.exp_config_name)

    # While loop until no any task
    while task["task_name"] != FINISH_EXECUTION and task is not None:
        if task["task_name"] == NO_READY_TASK:
            print('Sleeping for 10 seconds...', flush=True)
            time.sleep(10)  # Sleep for 10 seconds to wait for tasks to be unblocked
        else:
            # Use PipelineEvaluator to execute the task
            execution_status = pipeline_evaluator.execute_task(task_name=task["task_name"], seed=exp_config.random_state)
            if execution_status:
                virny_flow_client.complete_task(exp_config_name=exp_config.exp_config_name,
                                                task_guid=task["task_guid"],
                                                task_name=task["task_name"],
                                                stage_id=task["stage_id"])
            print('\n\n', flush=True)

        # Request a new task in VirnyFlow
        task = virny_flow_client.get_task(exp_config.exp_config_name)
