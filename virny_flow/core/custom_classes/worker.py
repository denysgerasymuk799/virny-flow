import os
import json

from dotenv import load_dotenv
from kafka import KafkaProducer, KafkaConsumer
from openbox.utils.history import Observation

from virny_flow.configs.constants import NEW_TASKS_QUEUE_TOPIC, COMPLETED_TASKS_QUEUE_TOPIC, WORKER_CONSUMER_GROUP
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.core.utils.pipeline_utils import observation_to_dict


def on_send_error(exc_info):
    print(f'ERROR Producer: Got errback -- {exc_info}')


class Worker:
    def __init__(self, address: str, secrets_path: str):
        load_dotenv(secrets_path, override=True)

        self.address = address.rstrip('/')
        self._logger = get_logger(logger_name="Worker")

        self.consumer = KafkaConsumer(
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
            request_timeout_ms=330_000,
        )
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
            while True:
                for msg_obj in self.consumer:
                    task_dct = msg_obj.value
                    if task_dct.get('task_uuid') is not None:
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

        self.producer.send(COMPLETED_TASKS_QUEUE_TOPIC, value=message).add_errback(on_send_error)
        self.producer.flush()
        self._logger.info(f"New task with UUID = {task_uuid} and run_num = {run_num} was completed and sent to COMPLETED_TASKS_QUEUE_TOPIC")
