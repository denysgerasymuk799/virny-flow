import os
import certifi
import motor.motor_asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from dataclasses import asdict
from pymongo import ASCENDING

from virny_flow.configs.structs import Task
from virny_flow.configs.constants import TASK_QUEUE_TABLE, TaskStatus, NO_TASKS


class TaskQueue:
    def __init__(self, secrets_path: str, max_queue_size: int):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env
        self.max_queue_size = max_queue_size

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

    async def has_space_for_next_lp(self, exp_config_name: str, num_workers: int):
        task_count = await self.collection.count_documents({
            "exp_config_name": exp_config_name,
            "task_status": { "$in": [TaskStatus.WAITING.value, TaskStatus.ASSIGNED.value] },
            "deletion_flag": False,
        })
        return task_count <= self.max_queue_size - num_workers

    async def is_empty(self, exp_config_name: str):
        task_count = await self.collection.count_documents({
            "exp_config_name": exp_config_name,
            "task_status": { "$in": [TaskStatus.WAITING.value, TaskStatus.ASSIGNED.value] },
            "deletion_flag": False,
        })
        return task_count == 0

    async def enqueue(self, task: Task):
        """Add an item to the queue."""
        task_record = asdict(task)
        task_record["task_status"] = TaskStatus.WAITING.value
        task_record["deletion_flag"] = False

        datetime_now = datetime.now(timezone.utc)
        task_record["create_datetime"] = datetime_now
        task_record["update_datetime"] = datetime_now

        await self.collection.insert_one(task_record)
        print(f"Enqueued task with UUID: {task.task_uuid}")

    async def dequeue(self, exp_config_name):
        """Remove a task from the queue."""
        # Find and update the first waiting item to processing status
        task = await self.collection.find_one_and_update(
            {"exp_config_name": exp_config_name, "task_status": TaskStatus.WAITING.value, "deletion_flag": False},
            {"$set": {"task_status": TaskStatus.ASSIGNED.value,
                      "update_datetime": datetime.now(timezone.utc)}},
            sort=[("_id", ASCENDING)]
        )
        if task:
            print(f"Dequeued task with UUID: {task['task_uuid']}")
            task["_id"] = str(task["_id"])
            task["create_datetime"] = str(task["create_datetime"])
            task["update_datetime"] = str(task["update_datetime"])
            return task
        else:
            print("Queue is empty.")
            return {"_id": None, "task_uuid": NO_TASKS}

    async def complete_task(self, exp_config_name: str, task_uuid: str):
        """Mark a task as completed."""
        resp = await self.collection.update_one(
            {"exp_config_name": exp_config_name, "task_uuid": task_uuid},
            {"$set": {"task_status": TaskStatus.DONE.value,
                      "update_datetime": datetime.now(timezone.utc)}}
        )
        print(f"Completed task with UUID: {task_uuid}")
        return resp.modified_count

    def close(self):
        self.client.close()
