import uvicorn
from fastapi import FastAPI
from munch import DefaultMunch

from .routes import register_routes
from .database.task_manager_db_client import TaskManagerDBClient
from virny_flow.configs.structs import BOAdvisorConfig


class TaskManager:
    def __init__(self, secrets_path: str, host: str, port: int, exp_config: DefaultMunch):
        self.host = host
        self.port = port
        self.exp_config = exp_config

        # Initialize BO Advisor
        self.bo_advisor_config = BOAdvisorConfig()
        self.bo_advisor_config.ref_point = exp_config.optimisation_args.ref_point
        self.bo_advisor_config.num_objectives = len(exp_config.optimisation_args.objectives)

        self.app = FastAPI()
        self.uvicorn_server = uvicorn.Server(uvicorn.Config(self.app, host=self.host, port=self.port))  # initialize to pass to register_routes()
        self.db_client = TaskManagerDBClient(secrets_path)

        # Separate MO-BO optimizer for each run_num and logical pipeline
        self._lp_to_advisor = {run_num: dict() for run_num in exp_config.common_args.run_nums}

        # Register routes from the routes module
        register_routes(app=self.app,
                        exp_config=self.exp_config,
                        db_client=self.db_client,
                        uvicorn_server=self.uvicorn_server,
                        lp_to_advisor=self._lp_to_advisor,
                        bo_advisor_config=self.bo_advisor_config)

        # Redefine uvicorn_server.config to initialize endpoints
        self.uvicorn_server.config = uvicorn.Config(self.app, host=self.host, port=self.port)

    def run(self):
        # The actual server runner using Uvicorn
        self.uvicorn_server.run()
