import os
import certifi
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient


class CoreDBClient:
    def __init__(self, secrets_path: str):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env

        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        self.connection_string = os.getenv("CONNECTION_STRING")
        self.db_name = os.getenv("DB_NAME")

        self.client = None

    def connect(self):
        # Create a connection using MongoClient
        self.client = MongoClient(self.connection_string, tlsCAFile=certifi.where())

    def _get_collection(self, collection_name):
        return self.client[self.db_name][collection_name]

    def execute_write_query(self, records, collection_name):
        collection = self._get_collection(collection_name)
        collection.insert_many(records)

    def update_query(self, collection_name: str, condition: dict, update_val_dct: dict, session = None):
        collection = self._get_collection(collection_name)
        condition['deletion_flag'] = False

        # Update many documents
        result = collection.update_many(
            condition,  # Filter to match the document
            {"$set": update_val_dct},  # Update operation
            session = session
        )
        return result.modified_count

    def read_one_query(self, collection_name: str, query: dict, sort_param: list = None, session = None):
        collection = self._get_collection(collection_name)
        query['deletion_flag'] = False
        return collection.find_one(query, sort=sort_param, session=session)

    def execute_read_query(self, collection_name: str, query: dict):
        collection = self._get_collection(collection_name)
        cursor = collection.find(query)
        records = []
        for record in cursor:
            del record['_id']
            records.append(record)

        return records

    def write_pandas_df_into_db(self, collection_name: str, df: pd.DataFrame, custom_tbl_fields_dct: dict = None):
        if df.shape[0] == 0:
            print('Dataframe is empty. Skip writing to a database.')
            return

        # Append custom fields to the df
        for column, value in custom_tbl_fields_dct.items():
            df[column] = value

        # Rename Pandas columns to lower case
        df.columns = df.columns.str.lower()
        df['deletion_flag'] = False

        self.execute_write_query(df.to_dict('records'), collection_name)
        print('Dataframe is successfully written into the database')

    def read_metric_df_from_db(self, collection_name: str, query: dict):
        records = self.execute_read_query(query=query,
                                          collection_name=collection_name)
        metric_df = pd.DataFrame(records)

        # Capitalize column names to be consistent across the whole library
        new_column_names = []
        for col in metric_df.columns:
            new_col_name = '_'.join([c.capitalize() for c in col.split('_')])
            new_column_names.append(new_col_name)

        metric_df.columns = new_column_names
        return metric_df

    def get_db_writer(self, collection_name: str):
        collection_obj = self._get_collection(collection_name)

        def db_writer_func(run_models_metrics_df, collection=collection_obj):
            # Rename Pandas columns to lower case
            run_models_metrics_df.columns = run_models_metrics_df.columns.str.lower()
            collection.insert_many(run_models_metrics_df.to_dict('records'))

        return db_writer_func

    def close(self):
        self.client.close()
