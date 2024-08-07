import os
import boto3
import pandas as pd

from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


class S3Client:
    def __init__(self, secrets_path: str):
        load_dotenv(secrets_path, override=True)  # Take environment variables from .env

        self.bucket_name = os.getenv("BUCKET_NAME")
        self.region_name = os.getenv("REGION_NAME")

        session = boto3.Session(
            aws_access_key_id=os.getenv("USER_PUBLIC_KEY"),
            aws_secret_access_key=os.getenv("USER_SECRET_KEY"),
            region_name=self.region_name
        )
        self.s3_client = session.client('s3')

    def write_csv(self, df: pd.DataFrame, key: str) -> bool:
        """
        Write a DataFrame as a CSV file to an S3 bucket.

        :param df: The DataFrame to be stored as a CSV.
        :param key: The S3 key (file name) under which the CSV will be stored.
        :return: True if the CSV was stored, otherwise False.
        """
        try:
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=csv_buffer.getvalue())
            print(f"CSV written to {self.bucket_name}/{key}")
            return True
        except NoCredentialsError:
            print("Credentials not available")
            return False
        except PartialCredentialsError:
            print("Incomplete credentials provided")
            return False

    def read_csv(self, key: str) -> pd.DataFrame:
        """
        Read a CSV file from an S3 bucket into a DataFrame.

        :param key: The S3 key (file name) of the CSV to be read.
        :return: The DataFrame if the CSV was found, otherwise None.
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            csv_buffer = BytesIO(response['Body'].read())
            df = pd.read_csv(csv_buffer)
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

    def list_files(self, prefix: str = '') -> list:
        """
        List files in the S3 bucket with the given prefix.

        :param prefix: Prefix to filter the files.
        :return: List of file keys.
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            files = [obj['Key'] for obj in response.get('Contents', [])]
            return files
        except NoCredentialsError:
            print("Credentials not available")
            return []
        except PartialCredentialsError:
            print("Incomplete credentials provided")
            return []
