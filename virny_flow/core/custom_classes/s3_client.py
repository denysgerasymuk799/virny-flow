import os
import json
import boto3
import pickle
import pandas as pd

from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from virny_flow.configs.constants import S3Folder


class S3Client:
    def __init__(self, secrets_path: str):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env

        self.root_path = S3Folder.virny_flow.value
        self.bucket_name = os.getenv("BUCKET_NAME")
        self.region_name = os.getenv("REGION_NAME")

        session = boto3.Session(
            aws_access_key_id=os.getenv("USER_PUBLIC_KEY"),
            aws_secret_access_key=os.getenv("USER_SECRET_KEY"),
            region_name=self.region_name
        )
        self.s3_client = session.client('s3')

    def list_files(self, prefix: str = '') -> list:
        """
        List files in the S3 bucket with the given prefix.

        :param prefix: Prefix to filter the files.
        :return: List of file keys.
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=f'{self.root_path}/{prefix}')
            files = [obj['Key'] for obj in response.get('Contents', [])]
            return files
        except NoCredentialsError:
            print("Credentials not available")
            return []
        except PartialCredentialsError:
            print("Incomplete credentials provided")
            return []

    def write_csv(self, df: pd.DataFrame, key: str, index: bool) -> bool:
        """
        Write a DataFrame as a CSV file to an S3 bucket.

        :param df: The DataFrame to be stored as a CSV.
        :param key: The S3 key (file name) under which the CSV will be stored.
        :return: True if the CSV was stored, otherwise False.
        """
        try:
            key = f'{self.root_path}/{key}'
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=index)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=csv_buffer.getvalue())
            print(f"CSV written to {self.bucket_name}/{key}")
            return True
        except NoCredentialsError:
            print("Credentials not available")
            return False
        except PartialCredentialsError:
            print("Incomplete credentials provided")
            return False

    def read_csv(self, key: str, index: bool) -> pd.DataFrame:
        """
        Read a CSV file from an S3 bucket into a DataFrame.

        :param key: The S3 key (file name) of the CSV to be read.
        :return: The DataFrame if the CSV was found, otherwise None.
        """
        try:
            key = f'{self.root_path}/{key}'
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            csv_buffer = BytesIO(response['Body'].read())
            df = pd.read_csv(csv_buffer, header=0, index_col=0 if index else None)
            print(f"CSV read from {self.bucket_name}/{key}")
            return df
        except self.s3_client.exceptions.NoSuchKey:
            print("The CSV does not exist")
            return None
        except NoCredentialsError:
            print("Credentials not available")
            return None
        except PartialCredentialsError:
            print("Incomplete credentials provided")
            return None

    def write_pickle(self, obj, key):
        """
        Saves a Python object as a pickle file to an S3 bucket.

        :param obj: The Python object to be pickled and saved.
        :param key: The S3 key (path) where the pickle file will be saved.
        """
        key = f'{self.root_path}/{key}'
        pickle_buffer = BytesIO()
        pickle.dump(obj, pickle_buffer)
        pickle_buffer.seek(0)

        self.s3_client.upload_fileobj(pickle_buffer, self.bucket_name, key)
        print(f"Pickle file was saved to s3://{self.bucket_name}/{key}")

    def read_pickle(self, key):
        """
        Loads a Python object from a pickle file in an S3 bucket.

        :param key: The S3 key (path) of the pickle file to load.
        :return: The unpickled Python object.
        """
        key = f'{self.root_path}/{key}'
        pickle_buffer = BytesIO()

        self.s3_client.download_fileobj(self.bucket_name, key, pickle_buffer)
        pickle_buffer.seek(0)

        obj = pickle.load(pickle_buffer)
        print(f"Pickle file was loaded from s3://{self.bucket_name}/{key}")
        return obj

    def write_json(self, data, key):
        """
        Saves a Python dictionary as a JSON file to an S3 bucket.

        :param data: The Python dictionary to be saved as JSON.
        :param key: The S3 key (path) where the JSON file will be saved.
        """
        key = f'{self.root_path}/{key}'
        # Convert the dictionary to a JSON string and then encode it to bytes
        json_buffer = BytesIO(json.dumps(data).encode('utf-8'))

        # Upload the JSON bytes to S3
        self.s3_client.upload_fileobj(json_buffer, self.bucket_name, key)
        print(f"JSON file saved to s3://{self.bucket_name}/{key}")

    def read_json(self, key):
        """
        Loads a Python dictionary from a JSON file in an S3 bucket.

        :param key: The S3 key (path) of the JSON file to load.
        :return: The Python dictionary loaded from the JSON file.
        """
        key = f'{self.root_path}/{key}'
        json_buffer = BytesIO()

        self.s3_client.download_fileobj(self.bucket_name, key, json_buffer)
        json_buffer.seek(0)

        data = json.load(json_buffer)
        print(f"JSON file loaded from s3://{self.bucket_name}/{key}")
        return data
