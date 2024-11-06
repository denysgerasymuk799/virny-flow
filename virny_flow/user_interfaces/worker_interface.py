import os
import warnings

from munch import DefaultMunch
from virny_flow.configs.constants import NO_TASKS
from virny_flow.configs.structs import Task
from virny_flow.core.custom_classes.pipeline_evaluator import PipelineEvaluator
from virny_flow.core.custom_classes.worker import Worker


def worker_interface(exp_config: DefaultMunch, virny_flow_address: str, dataset_config: dict,
                     null_imputation_config: dict, fairness_intervention_config: dict, models_config: dict):
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Init objects
    pipeline_evaluator = PipelineEvaluator(exp_config=exp_config,
                                           dataset_config=dataset_config,
                                           null_imputation_config=null_imputation_config,
                                           fairness_intervention_config=fairness_intervention_config,
                                           models_config=models_config)

    # Get an initial task
    worker = Worker(address=virny_flow_address)
    task_dct = worker.get_task(exp_config.exp_config_name)

    # Infinite while loop for task execution
    while True:
        if task_dct["task_uuid"] == NO_TASKS:
            # Shutdown the worker to know when all job is done.
            # Can be useful when the job sends you an email when it is done.
            print('Queue is empty. Shutting down the worker...', flush=True)
            return
        else:
            task = Task.from_dict(task_dct)

            # Use PipelineEvaluator to execute the task
            observation = pipeline_evaluator.execute_task(task=task, seed=exp_config.random_state)
            if observation:
                worker.complete_task(exp_config_name=task.exp_config_name,
                                     task_uuid=task.task_uuid,
                                     logical_pipeline_uuid=task.physical_pipeline.logical_pipeline_uuid,
                                     logical_pipeline_name=task.physical_pipeline.logical_pipeline_name,
                                     observation=observation)
            print('\n\n', flush=True)

        # Request a new task in VirnyFlow
        task_dct = worker.get_task(exp_config.exp_config_name)
