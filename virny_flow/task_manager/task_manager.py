import uvicorn
import threading
from fastapi import FastAPI

from routes import register_routes
from database.database_client import DatabaseClient


class TaskManager:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.db_client = DatabaseClient()

        # Register routes from the routes module
        register_routes(self.app)

    def run(self):
        """Run the FastAPI web server in a background thread."""
        thread = threading.Thread(target=self._run_server, daemon=True)
        thread.start()
        print(f"TaskManager is running on {self.host}:{self.port}")

    def _run_server(self):
        """The actual server runner using Uvicorn."""
        uvicorn.run(self.app, host=self.host, port=self.port)
