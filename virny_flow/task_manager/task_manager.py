import uvicorn
from fastapi import FastAPI
from munch import DefaultMunch

from .routes import register_routes
from .database.task_manager_db_client import TaskManagerDBClient
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.configs.structs import BOAdvisorConfig


class TaskManager:
    def __init__(self, secrets_path: str, host: str, port: int, exp_config: DefaultMunch,
                 bo_advisor_config: BOAdvisorConfig):
        self.host = host
        self.port = port
        self.exp_config = exp_config
        self.bo_advisor_config = bo_advisor_config

        self.app = FastAPI()
        self.db_client = TaskManagerDBClient(secrets_path)
        self._logger = get_logger(logger_name="task_manager")
        self._lp_to_advisor = dict() # Separate MO-BO optimizer for each logical pipeline

        # Register routes from the routes module
        register_routes(app=self.app,
                        exp_config=self.exp_config,
                        db_client=self.db_client,
                        lp_to_advisor=self._lp_to_advisor,
                        bo_advisor_config=self.bo_advisor_config,
                        logger=self._logger)

    def run(self):
        """The actual server runner using Uvicorn."""
        uvicorn.run(self.app, host=self.host, port=self.port)
