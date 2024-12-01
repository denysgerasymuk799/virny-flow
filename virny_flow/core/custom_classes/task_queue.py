import os
import certifi
import motor.motor_asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from dataclasses import asdict
from pymongo import ASCENDING

from virny_flow.configs.structs import Task
from virny_flow.configs.constants import TASK_QUEUE_TABLE, TaskStatus, NO_TASKS
from virny_flow.core.utils.custom_logger import get_logger


class TaskQueue:
    def __init__(self, secrets_path: str, max_queue_size: int):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env
        self.max_queue_size = max_queue_size
        self._logger = get_logger('TaskQueue')

        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        self.connection_string = os.getenv("CONNECTION_STRING")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = TASK_QUEUE_TABLE

        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string,
                                                             serverSelectionTimeoutMS=60_000,
                                                             tls=True,
                                                             tlsAllowInvalidCertificates=True,
                                                             tlsCAFile=certifi.where())

        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

        # Create an index to ensure atomic dequeue operation.
        # Note, if an index with the same fields and options already exists,
        # MongoDB will skip re-creating it automatically.
        self.collection.create_index([("task_status", ASCENDING), ("_id", ASCENDING)])

    def _get_condition(self, condition: dict, exp_config_name: str, run_num: int):
        condition["exp_config_name"] = exp_config_name
        condition["deletion_flag"] = False
        if run_num is not None:
            condition["run_num"] = run_num
        return condition

    async def has_space_for_next_lp(self, exp_config_name: str, num_pp_candidates: int, run_num: int):
        condition = {
            "task_status": { "$in": [TaskStatus.WAITING.value, TaskStatus.ASSIGNED.value] },
        }
        condition = self._get_condition(condition=condition,
                                        exp_config_name=exp_config_name,
                                        run_num=run_num)
        task_count = await self.collection.count_documents(condition)

        return task_count <= self.max_queue_size - num_pp_candidates

    async def is_empty(self, exp_config_name: str, run_num: int):
        condition = {
            "task_status": { "$in": [TaskStatus.WAITING.value, TaskStatus.ASSIGNED.value] },
        }
        condition = self._get_condition(condition=condition,
                                        exp_config_name=exp_config_name,
                                        run_num=run_num)
        task_count = await self.collection.count_documents(condition)

        return task_count == 0

    async def get_num_available_tasks(self, exp_config_name: str, run_num: int):
        condition = {"task_status": TaskStatus.WAITING.value}
        condition = self._get_condition(condition=condition,
                                        exp_config_name=exp_config_name,
                                        run_num=run_num)
        task_count = await self.collection.count_documents(condition)

        return task_count

    async def enqueue(self, task: Task):
        """Add an item to the queue."""
        task_record = asdict(task)
        task_record["task_status"] = TaskStatus.WAITING.value
        task_record["deletion_flag"] = False

        datetime_now = datetime.now(timezone.utc)
        task_record["create_datetime"] = datetime_now
        task_record["update_datetime"] = datetime_now

        await self.collection.insert_one(task_record)
        self._logger.info(f"Enqueued task with UUID: {task.task_uuid}")

    async def dequeue(self, exp_config_name: str, run_num: int):
        """Remove a task from the queue."""
        condition = {"task_status": TaskStatus.WAITING.value}
        condition = self._get_condition(condition=condition,
                                        exp_config_name=exp_config_name,
                                        run_num=run_num)

        # Find and update the first waiting item to processing status
        task = await self.collection.find_one_and_update(
            condition,
            {"$set": {"task_status": TaskStatus.ASSIGNED.value,
                      "update_datetime": datetime.now(timezone.utc)}},
            sort=[("_id", ASCENDING), ("run_num", ASCENDING)]
        )
        if task:
            self._logger.info(f"Dequeued task with UUID: {task['task_uuid']}")
            task["_id"] = str(task["_id"])
            task["create_datetime"] = str(task["create_datetime"])
            task["update_datetime"] = str(task["update_datetime"])
            return task
        else:
            self._logger.info("Queue is empty.")
            return {"_id": None, "task_uuid": NO_TASKS}

    async def complete_task(self, exp_config_name: str, task_uuid: str, run_num: int):
        condition = {"task_uuid": task_uuid}
        condition = self._get_condition(condition=condition,
                                        exp_config_name=exp_config_name,
                                        run_num=run_num)

        """Mark a task as completed."""
        resp = await self.collection.update_one(
            condition,
            {"$set": {"task_status": TaskStatus.DONE.value,
                      "update_datetime": datetime.now(timezone.utc)}}
        )
        self._logger.info(f"Completed task with UUID: {task_uuid}")
        return resp.modified_count

    def close(self):
        self.client.close()
