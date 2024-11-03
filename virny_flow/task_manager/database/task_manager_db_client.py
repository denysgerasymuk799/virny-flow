import os
import pathlib
import certifi
import motor.motor_asyncio

from bson.objectid import ObjectId
from dotenv import load_dotenv
from datetime import datetime, timezone

from virny_flow.configs.constants import (EXP_PROGRESS_TRACKING_TABLE, FINISH_EXECUTION, NO_READY_TASK,
                                          TaskStatus, STAGE_NAME_TO_STAGE_ID, StageName)


class TaskManagerDBClient:
    def __init__(self, secrets_path: str = pathlib.Path(__file__).parent.parent.joinpath( 'configs', 'secrets.env')):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env

        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        self.connection_string = os.getenv("CONNECTION_STRING")
        self.db_name = os.getenv("DB_NAME")

        self.client = None

    def connect(self):
        # Create a connection using MongoClient
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string,
                                                             serverSelectionTimeoutMS=60_000,
                                                             tls=True,
                                                             tlsAllowInvalidCertificates=True,
                                                             tlsCAFile=certifi.where())

    def _get_collection(self, collection_name):
        return self.client[self.db_name][collection_name]

    async def check_record_exists(self, query: dict):
        collection = self._get_collection(collection_name=EXP_PROGRESS_TRACKING_TABLE)
        query['tag'] = 'OK'
        return await collection.find_one(query) is not None

    async def execute_write_query(self, records, collection_name):
        collection = self._get_collection(collection_name)
        await collection.insert_many(records)

    async def update_one_query(self, collection_name: str, _id, update_val_dct: dict):
        collection = self._get_collection(collection_name)
        object_id = ObjectId(_id) if isinstance(_id, str) else _id

        # Update a single document
        result = await collection.update_one(
            {"_id": object_id},  # Filter to match the document
            {"$set": update_val_dct}  # Update operation
        )
        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")
        return result.modified_count

    async def update_query(self, collection_name: str, condition: dict, update_val_dct: dict):
        collection = self._get_collection(collection_name)
        condition['tag'] = 'OK'

        # Update many documents
        result = await collection.update_many(
            condition,  # Filter to match the document
            {"$set": update_val_dct}  # Update operation
        )
        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")
        return result.modified_count

    async def read_one_query(self, collection_name: str, query: dict, sort_param: list = None):
        collection = self._get_collection(collection_name)
        query['tag'] = 'OK'
        return await collection.find_one(query, sort=sort_param)

    async def count_query(self, collection_name: str, condition: dict):
        collection = self._get_collection(collection_name)
        condition['tag'] = 'OK'
        return await collection.count_documents(condition)

    async def write_records_into_db(self, collection_name: str, records: list, static_values_dct: dict):
        for key, value in static_values_dct.items():
            for record in records:
                record[key] = value

        await self.execute_write_query(records, collection_name)

    async def read_worker_task_from_db(self, exp_config_name: str):
        collection_name = EXP_PROGRESS_TRACKING_TABLE
        query = {
            'exp_config_name': exp_config_name,
            'task_status': TaskStatus.READY.value,
        }

        high_priority_task = await self.read_one_query(collection_name, query, sort_param=[('task_id', 1)])
        if high_priority_task is not None:
            await self.update_one_query(collection_name, _id=high_priority_task["_id"],
                                        update_val_dct={"task_status": TaskStatus.ASSIGNED.value,
                                                        "update_datetime": datetime.now(timezone.utc)})
            return high_priority_task
        else:
            num_blocked_tasks = await self.count_query(collection_name, condition={'exp_config_name': exp_config_name,
                                                                                   'task_status': TaskStatus.BLOCKED.value})
            num_ready_tasks = await self.count_query(collection_name, condition={'exp_config_name': exp_config_name,
                                                                                 'task_status': TaskStatus.READY.value})
            if num_blocked_tasks + num_ready_tasks > 0:
                return {"_id": None, "task_name": NO_READY_TASK, "stage_id": None}
            
            return {"_id": None, "task_name": FINISH_EXECUTION, "stage_id": None}

    async def complete_worker_task_in_db(self, exp_config_name: str, task_guid: str, task_name: str, stage_id: int):
        # Set the current task as DONE
        done_tasks_count = await self.update_one_query(collection_name=EXP_PROGRESS_TRACKING_TABLE,
                                                       _id=task_guid,
                                                       update_val_dct={"task_status": TaskStatus.DONE.value,
                                                                       "update_datetime": datetime.now(timezone.utc)})
        # Unblock further related tasks if exist
        ready_tasks_count = None
        if stage_id != STAGE_NAME_TO_STAGE_ID[StageName.model_evaluation.value]:
            ready_tasks_count = await self.update_query(collection_name=EXP_PROGRESS_TRACKING_TABLE,
                                                        condition={"exp_config_name": exp_config_name,
                                                                   "stage_id": stage_id + 1,
                                                                   "task_name": {"$regex": f'^{task_name}'}},
                                                        update_val_dct={"task_status": TaskStatus.READY.value,
                                                                        "update_datetime": datetime.now(timezone.utc)})
        return done_tasks_count, ready_tasks_count

    def close(self):
        self.client.close()
