from virny_flow.configs.constants import NO_TASKS
from virny_flow.utils.custom_logger import get_logger
from virny_flow.custom_classes.web_server_client import WebServerClient


def worker_interface(exp_config_name: str, virny_flow_address: str):
    logger = get_logger()

    # Get an initial task
    web_server_client = WebServerClient(address=virny_flow_address)
    task = web_server_client.get_worker_task(exp_config_name)

    # While loop until no any task
    while task["task_name"] != NO_TASKS and task is not None:
        # Use Benchmark to execute the task

        logger.info(f'Task {task["task_name"]} was executed')
        web_server_client.complete_worker_task(task_id=task["task_id"], task_name=task["task_name"])

        # Request a new task in VirnyFlow
        task = web_server_client.get_worker_task(exp_config_name)
