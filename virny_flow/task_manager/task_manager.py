import uvicorn
import threading
from fastapi import FastAPI

from .routes import register_routes
from .database.task_manager_db_client import TaskManagerDBClient
from virny_flow.core.utils.custom_logger import get_logger


class TaskManager:
    def __init__(self, secrets_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.db_client = TaskManagerDBClient(secrets_path)
        self._logger = get_logger(logger_name="task_manager")

        # Register routes from the routes module
        register_routes(app=self.app, db_client=self.db_client, logger=self._logger)

    def run(self):
        """The actual server runner using Uvicorn."""
        uvicorn.run(self.app, host=self.host, port=self.port)
