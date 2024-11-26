import os
import time
import json
from munch import DefaultMunch
from datetime import datetime, timezone
from openbox.utils.history import Observation
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .score_model import update_logical_pipeline_score_model
from .bayesian_optimization import parse_config_space
from ..config import logger
from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.core.custom_classes.task_queue import TaskQueue
from virny_flow.configs.constants import (TASK_MANAGER_CONSUMER_GROUP, NEW_TASKS_QUEUE_TOPIC,
                                          COMPLETED_TASKS_QUEUE_TOPIC, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE)


async def start_task_provider(exp_config: DefaultMunch, task_queue: TaskQueue):
    restart_producer = False
    print('KAFKA_BROKER --', os.getenv("KAFKA_BROKER"))
    producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BROKER"))
    await producer.start()

    try:
        while True:
            num_available_tasks = await task_queue.get_num_available_tasks(exp_config_name=exp_config.exp_config_name)
            if num_available_tasks == 0:
                print("Wait until new tasks come up...")
                time.sleep(10)
                continue

            # Add new tasks to the NewTaskQueue topic in Kafka
            for _ in range(num_available_tasks):
                new_high_priority_task = await task_queue.dequeue(exp_config_name=exp_config.exp_config_name)
                json_message = json.dumps(new_high_priority_task)  # Serialize to JSON
                logger.info(f'New task was retrieved, UUID: {new_high_priority_task["task_uuid"]}')

                # In case the producer was stopped due to an error
                if restart_producer:
                    await producer.start()  # Get cluster layout and initial topic/partition leadership information
                    restart_producer = False

                try:
                    await producer.send_and_wait(NEW_TASKS_QUEUE_TOPIC, json_message.encode('utf-8'))
                except Exception as e:
                    logger.info(f'Sending message to Kafka failed due to the following error -- {e}')
                    # Wait for all pending messages to be delivered or expire.
                    await producer.stop()
                    restart_producer = True
    finally:
        await producer.stop()


async def start_cost_model_updater(exp_config: DefaultMunch, lp_to_advisor: dict,
                                   db_client: TaskManagerDBClient, task_queue: TaskQueue):
    consumer = AIOKafkaConsumer(COMPLETED_TASKS_QUEUE_TOPIC,
                                bootstrap_servers=[os.getenv("KAFKA_BROKER")],
                                group_id=TASK_MANAGER_CONSUMER_GROUP,
                                auto_offset_reset="latest")
    await consumer.start()
    try:
        async for record in consumer:
            data = json.loads(record.value)
            print(f'Consumer record: {data["task_uuid"]}')

            # Process body
            exp_config_name = data["exp_config_name"]
            task_uuid = data["task_uuid"]
            physical_pipeline_uuid = data["physical_pipeline_uuid"]
            logical_pipeline_uuid = data["logical_pipeline_uuid"]
            logical_pipeline_name = data["logical_pipeline_name"]

            data["observation"]["config"] = parse_config_space(data["observation"]["config"])
            observation = Observation.from_dict(data["observation"],
                                                config_space=lp_to_advisor[logical_pipeline_name]["config_space"])

            # Update the advisor of the logical pipeline
            lp_to_advisor[logical_pipeline_name]["config_advisor"].update_observation(observation)

            # Add an observation to DB
            datetime_now = datetime.now(timezone.utc)
            await db_client.write_records_into_db(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                                  records=[observation.to_dict()],
                                                  exp_config_name=exp_config_name,
                                                  static_values_dct={
                                                      "task_uuid": task_uuid,
                                                      "physical_pipeline_uuid": physical_pipeline_uuid,
                                                      "logical_pipeline_uuid": logical_pipeline_uuid,
                                                      "logical_pipeline_name": logical_pipeline_name,
                                                      "create_datetime": datetime_now,
                                                      "update_datetime": datetime_now,
                                                  })
            # Update score of the selected logical pipeline
            await update_logical_pipeline_score_model(exp_config_name=exp_config_name,
                                                      objectives_lst=exp_config.objectives,
                                                      logical_pipeline_uuid=logical_pipeline_uuid,
                                                      db_client=db_client)

            # Complete the task
            await task_queue.complete_task(exp_config_name=exp_config_name, task_uuid=task_uuid)
            logger.info(f'Task with task_uuid = {task_uuid} was successfully completed.')

            await consumer.commit()
    except Exception as err:
        logger.error(f'Consume error: {err}')
    finally:
        await consumer.stop()
