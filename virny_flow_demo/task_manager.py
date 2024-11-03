import os
from virny_flow.task_manager import TaskManager


if __name__ == "__main__":
    task_manager = TaskManager(secrets_path=os.path.join(os.getcwd(), "configs", "secrets.env"))
    task_manager.run()
