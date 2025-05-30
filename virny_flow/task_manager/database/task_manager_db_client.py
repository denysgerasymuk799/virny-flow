import os
import certifi
import motor.motor_asyncio

from datetime import datetime, timezone
from bson.objectid import ObjectId
from dotenv import load_dotenv

from virny_flow.configs.constants import EXP_CONFIG_HISTORY_TABLE


class TaskManagerDBClient:
    def __init__(self, secrets_path: str):
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
        collection = self._get_collection(collection_name=EXP_CONFIG_HISTORY_TABLE)
        query['deletion_flag'] = False
        return await collection.find_one(query) is not None

    async def upsert_query(self, collection_name: str, exp_config_name: str, run_num: int,
                           condition: dict, record: dict):
        """
        Inserts a record if it does not exist, or updates it if it exists.
        """
        collection = self._get_collection(collection_name)
        condition["exp_config_name"] = exp_config_name
        condition["run_num"] = run_num
        condition['deletion_flag'] = False

        record["exp_config_name"] = exp_config_name
        record["run_num"] = run_num
        record["update_datetime"] = datetime.now(timezone.utc)
        record['deletion_flag'] = False

        # Update many documents
        result = await collection.update_many(
            condition,           # Match the record using this query
            {"$set": record},  # Update fields with this data
            upsert=True             # Insert if not found
        )

        if result.upserted_id:
            print(f"Inserted a new record with ID: {result.upserted_id}")
        else:
            print("Updated existing record.")

        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": result.upserted_id,
        }

    async def execute_write_query(self, records: list, collection_name: str):
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

    async def update_query(self, collection_name: str, condition: dict, update_val_dct: dict,
                           exp_config_name: str, run_num: int):
        collection = self._get_collection(collection_name)
        condition["exp_config_name"] = exp_config_name
        condition["run_num"] = run_num
        condition['deletion_flag'] = False

        # Update many documents
        result = await collection.update_many(
            condition,  # Filter to match the document
            {"$set": update_val_dct}  # Update operation
        )
        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")
        return result.modified_count

    async def delete_query(self, collection_name: str, condition: dict, exp_config_name: str, run_num: int):
        collection = self._get_collection(collection_name)
        condition['exp_config_name'] = exp_config_name
        condition['run_num'] = run_num
        condition['tag'] = 'OK'
        result = await collection.delete_many(condition)
        print(f"Deleted {result.deleted_count} document(s).")

        return result.deleted_count

    async def increment_query(self, collection_name: str, condition: dict, increment_val_dct: dict,
                              exp_config_name: str, run_num: int):
        collection = self._get_collection(collection_name)
        condition['exp_config_name'] = exp_config_name
        condition['run_num'] = run_num
        condition['deletion_flag'] = False

        # Update many documents
        result = await collection.update_many(
            condition,  # Filter to match the document
            {"$inc": increment_val_dct}  # Update operation
        )
        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")

        return result.modified_count

    async def read_one_query(self, collection_name: str, query: dict, exp_config_name: str,
                             run_num: int, sort_param: list = None):
        collection = self._get_collection(collection_name)
        query['exp_config_name'] = exp_config_name
        query['run_num'] = run_num
        query['deletion_flag'] = False
        return await collection.find_one(query, sort=sort_param)

    async def find_many_documents(self, collection_name: str, query: dict, exp_config_name: str, run_num: int,
                                  sort_param: list = None, projection: dict = None):
        query['exp_config_name'] = exp_config_name
        query['run_num'] = run_num
        query['deletion_flag'] = False

        collection = self._get_collection(collection_name)
        cursor = collection.find(query, sort=sort_param, projection=projection)
        documents = []
        async for document in cursor:
            documents.append(document)
        return documents

    async def read_query(self, collection_name: str, query: dict, exp_config_name: str, run_num: int,
                         sort_param: list = None, projection: dict = None):
        return await self.find_many_documents(collection_name=collection_name,
                                              exp_config_name=exp_config_name,
                                              run_num=run_num,
                                              query=query,
                                              sort_param=sort_param,
                                              projection=projection)

    async def count_query(self, collection_name: str, condition: dict, exp_config_name: str, run_num: int):
        collection = self._get_collection(collection_name)
        condition['exp_config_name'] = exp_config_name
        condition['run_num'] = run_num
        condition['deletion_flag'] = False
        return await collection.count_documents(condition)

    async def write_records_into_db(self, collection_name: str, records: list, static_values_dct: dict,
                                    exp_config_name: str, run_num: int):
        static_values_dct["exp_config_name"] = exp_config_name
        static_values_dct["run_num"] = run_num
        static_values_dct["deletion_flag"] = False # By default, mark new records with deletion_flag = False
        for key, value in static_values_dct.items():
            for record in records:
                record[key] = value

        await self.execute_write_query(records, collection_name)

    def close(self):
        self.client.close()
