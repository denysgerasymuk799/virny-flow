import uvicorn
from fastapi import FastAPI
from munch import DefaultMunch

from .routes import register_routes
from .database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.custom_classes.task_queue import TaskQueue


class TaskManager:
    def __init__(self, secrets_path: str, host: str, port: int, exp_config: DefaultMunch):
        self.host = host
        self.port = port
        self.exp_config = exp_config

        # Initialize BO Advisor
        self.bo_advisor_config = BOAdvisorConfig()
        self.bo_advisor_config.ref_point = exp_config.ref_point
        self.bo_advisor_config.num_objectives = len(exp_config.objectives)

        self.app = FastAPI()
        self.uvicorn_server = None  # initialize to pass to register_routes()
        self.db_client = TaskManagerDBClient(secrets_path)
        self.task_queue = TaskQueue(secrets_path=secrets_path,
                                    max_queue_size=exp_config.queue_size)
        # Separate MO-BO optimizer for each run_num and logical pipeline
        self._lp_to_advisor = {run_num: dict() for run_num in exp_config.run_nums}

        # Register routes from the routes module
        register_routes(app=self.app,
                        exp_config=self.exp_config,
                        db_client=self.db_client,
                        uvicorn_server=self.uvicorn_server,
                        task_queue=self.task_queue,
                        lp_to_advisor=self._lp_to_advisor,
                        bo_advisor_config=self.bo_advisor_config)

        # Start server after defining endpoints
        self.uvicorn_server = uvicorn.Server(uvicorn.Config(self.app, host=self.host, port=self.port))

    def run(self):
        # The actual server runner using Uvicorn
        self.uvicorn_server.run()
