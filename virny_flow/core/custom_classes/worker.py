import os
import json
import time

from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer
from openbox.utils.history import Observation

from virny_flow.configs.constants import NEW_TASKS_QUEUE_TOPIC, COMPLETED_TASKS_QUEUE_TOPIC, WORKER_CONSUMER_GROUP
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.core.utils.pipeline_utils import observation_to_dict


def on_send_error(exc_info):
    print(f'ERROR Producer: Got errback -- {exc_info}')


def initialize_consumer():
    max_retries = 5  # Maximum number of retries
    backoff_factor = 2  # Exponential backoff factor (e.g., 2, 4, 8 seconds)
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1} to initialize Kafka consumer...")

            consumer = KafkaConsumer(
                NEW_TASKS_QUEUE_TOPIC,
                group_id=WORKER_CONSUMER_GROUP,
                bootstrap_servers=os.getenv("KAFKA_BROKER"),
                api_version=(0, 10, 1),
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                enable_auto_commit=True,
                auto_offset_reset="earliest",
                session_timeout_ms=300_000,  # Increase session timeout (default: 10000 ms)
                heartbeat_interval_ms=20_000,  # Increase heartbeat interval (default: 3000 ms)
                max_poll_interval_ms=600_000, # Up to 10 minutes to process a batch of messages
                max_poll_records=1,
                request_timeout_ms=330_000,
            )
            print("Kafka consumer initialized successfully.")
            return consumer  # Return the initialized consumer

        except Exception as e:
            print('Consumer initialization exception:', e)
            retry_count += 1
            print(f"Failed to initialize Kafka consumer: {e}")
            if retry_count < max_retries:
                wait_time = backoff_factor ** retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Exceeded maximum retries. Aborting.")
                raise  # Re-raise the exception if all retries fail


class Worker:
    def __init__(self, address: str, secrets_path: str):
        load_dotenv(secrets_path, override=True)

        self.address = address.rstrip('/')
        self._logger = get_logger(logger_name="Worker")

        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BROKER"),
            api_version=(0, 10, 1),
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    def get_task(self):
        try:
            # Since an execution of one task can take hours, it is better to re-initialize a consumer to get each new task
            consumer = initialize_consumer()
            while True:
                for msg_obj in consumer:
                    task_dct = msg_obj.value
                    if task_dct.get('task_uuid') is not None:
                        consumer.close()
                        self._logger.info(f"New task with UUID {task_dct['task_uuid']} was taken.")
                        return task_dct

        except ValueError as e:
            self._logger.error(f'Failed to retrieve a new task. Error occurred: {e}')


    def complete_task(self, exp_config_name: str, run_num: int, task_uuid: str, physical_pipeline_uuid: str,
                      logical_pipeline_uuid: str, logical_pipeline_name: str, observation: Observation):
        message = {
            "exp_config_name": exp_config_name,
            "run_num": run_num,
            "task_uuid": task_uuid,
            "physical_pipeline_uuid": physical_pipeline_uuid,
            "logical_pipeline_uuid": logical_pipeline_uuid,
            "logical_pipeline_name": logical_pipeline_name,
            "observation": observation_to_dict(observation),
        }

        # Send a message with retries
        max_retries = 3
        backoff_factor = 2
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Attempt to send the message
                self.producer.send(COMPLETED_TASKS_QUEUE_TOPIC, value=message).add_errback(on_send_error)
                self.producer.flush()
                self._logger.info(f"New task with UUID = {task_uuid} and run_num = {run_num} was completed and sent to COMPLETED_TASKS_QUEUE_TOPIC")
                break  # Exit the loop if sending is successful
            except Exception as e:
                retry_count += 1
                self._logger.error(f"Failed to send a message: {e}")
                if retry_count < max_retries:
                    wait_time = backoff_factor ** retry_count
                    self._logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self._logger.critical("Exceeded maximum retries. Message delivery failed.")
