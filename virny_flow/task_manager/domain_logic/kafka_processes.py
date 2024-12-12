import os
import time
import json
from munch import DefaultMunch
from datetime import datetime, timezone
from openbox.utils.history import Observation
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .score_model import update_logical_pipeline_score_model
from .bayesian_optimization import parse_config_space
from ..database.task_manager_db_client import TaskManagerDBClient
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.core.custom_classes.task_queue import TaskQueue
from virny_flow.configs.constants import (TASK_MANAGER_CONSUMER_GROUP, NEW_TASKS_QUEUE_TOPIC, LOGICAL_PIPELINE_SCORES_TABLE, NO_TASKS,
                                          EXP_CONFIG_HISTORY_TABLE, COMPLETED_TASKS_QUEUE_TOPIC, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE)


async def start_task_provider(exp_config: DefaultMunch, db_client: TaskManagerDBClient, task_queue: TaskQueue,
                              uvicorn_server):
    termination_flag = False
    logger = get_logger('TaskProvider')
    producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BROKER"))

    await producer.start()
    for idx, run_num in enumerate(exp_config.common_args.run_nums):
        execution_start_time = time.time()
        print('#' * 40 + '\n' + f'START TASK PROVIDING FOR RUN_NUM={run_num}' + '\n' + '#' * 40, flush=True)

        while True:
            try:
                num_available_tasks = await task_queue.get_num_available_tasks(exp_config_name=exp_config.common_args.exp_config_name, run_num=run_num)
                if num_available_tasks == 0:
                    if not await task_queue.is_empty(exp_config_name=exp_config.common_args.exp_config_name, run_num=run_num):
                        print("Wait until new tasks come up...", flush=True)
                        time.sleep(10)
                        continue

                    # Check termination factor: Get all logical pipelines, which have num_trials less than max_trials.
                    query = {"num_trials": {"$lt": exp_config.optimisation_args.max_trials}}
                    logical_pipeline_records = await db_client.read_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                                          exp_config_name=exp_config.common_args.exp_config_name,
                                                                          run_num=run_num,
                                                                          query=query)
                    if len(logical_pipeline_records) == 0:
                        # Save execution time of all pipelines for the define experimental config
                        exp_config_execution_time = time.time() - execution_start_time
                        await db_client.update_query(collection_name=EXP_CONFIG_HISTORY_TABLE,
                                                     exp_config_name=exp_config.common_args.exp_config_name,
                                                     run_num=run_num,
                                                     condition={},
                                                     update_val_dct={"exp_config_execution_time": exp_config_execution_time})
                        if idx + 1 == len(exp_config.common_args.run_nums):
                            # If all work is done, set num_available_tasks to num_workers and get new tasks from task_queue.
                            # When task_queue is empty, it will return NO_TASKS, and it will be sent to each worker.
                            num_available_tasks = exp_config.optimisation_args.num_workers * 2  # Multiply by 2 to be sure to shutdown all the workers
                            termination_flag = True
                        else:
                            break  # Start processing tasks for another run_num
                    else:
                        logger.info("Wait until all physical pipelines are executed to terminate all workers...")
                        time.sleep(10)
                        continue

                # Add new tasks to the NewTaskQueue topic in Kafka
                for i in range(num_available_tasks):
                    new_high_priority_task = await task_queue.dequeue(exp_config_name=exp_config.common_args.exp_config_name, run_num=run_num)
                    json_message = json.dumps(new_high_priority_task)  # Serialize to JSON
                    logger.info(f'New task was retrieved, UUID: {new_high_priority_task["task_uuid"]}')

                    try:
                        await producer.send_and_wait(topic=NEW_TASKS_QUEUE_TOPIC,
                                                     value=json_message.encode('utf-8'))
                    except Exception as e:
                        logger.info(f'Sending message to Kafka failed due to the following error -- {e}')
                        # Wait for all pending messages to be delivered or expire
                        await producer.stop()
                        producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BROKER"))
                        await producer.start()

                if termination_flag:
                    # Terminate CostModelUpdater
                    terminate_consumer_msg = {"exp_config_name": exp_config.common_args.exp_config_name, "run_num": run_num, "task_uuid": NO_TASKS}
                    terminate_consumer_json_msg = json.dumps(terminate_consumer_msg)
                    await producer.send_and_wait(topic=COMPLETED_TASKS_QUEUE_TOPIC,
                                                 value=terminate_consumer_json_msg.encode('utf-8'))
                    try:
                        # Terminate TaskManager web-server
                        uvicorn_server.should_exit = True
                    except Exception as e:
                        logger.error(f'Error during uvicorn server termination: {e}')

                    # Terminate TaskProvider
                    break

            except Exception as err:
                logger.error(f'Produce error: {err}')
                await producer.stop()
                producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BROKER"))
                await producer.start()

    await producer.stop()


def get_kafka_consumer():
    return AIOKafkaConsumer(COMPLETED_TASKS_QUEUE_TOPIC,
                            bootstrap_servers=[os.getenv("KAFKA_BROKER")],
                            group_id=TASK_MANAGER_CONSUMER_GROUP,
                            session_timeout_ms=300_000,  # Increase session timeout (default: 10000 ms)
                            heartbeat_interval_ms=20_000,  # Increase heartbeat interval (default: 3000 ms)
                            max_poll_interval_ms=600_000,  # Increase to 10 minutes if needed
                            max_poll_records=1,
                            request_timeout_ms=330_000,
                            auto_offset_reset="earliest",
                            enable_auto_commit=True)


async def start_cost_model_updater(exp_config: DefaultMunch, lp_to_advisor: dict,
                                   db_client: TaskManagerDBClient, task_queue: TaskQueue):
    logger = get_logger('CostModelUpdater')
    consumer = get_kafka_consumer()
    await consumer.start()
    try:
        async for record in consumer:
            try:
                data = json.loads(record.value)
                logger.info(f'Consumed a new task for cost model updating: {data["task_uuid"]}')

                # Process body
                exp_config_name = data["exp_config_name"]
                run_num = data["run_num"]
                task_uuid = data["task_uuid"]
                if task_uuid == NO_TASKS:
                    break

                physical_pipeline_uuid = data["physical_pipeline_uuid"]
                logical_pipeline_uuid = data["logical_pipeline_uuid"]
                logical_pipeline_name = data["logical_pipeline_name"]

                data["observation"]["config"] = parse_config_space(data["observation"]["config"])
                observation = Observation.from_dict(data["observation"],
                                                    config_space=lp_to_advisor[run_num][logical_pipeline_name]["config_space"])

                # Update the advisor of the logical pipeline
                lp_to_advisor[run_num][logical_pipeline_name]["config_advisor"].update_observation(observation)

                # Add an observation to DB
                datetime_now = datetime.now(timezone.utc)
                await db_client.write_records_into_db(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                                      records=[observation.to_dict()],
                                                      exp_config_name=exp_config_name,
                                                      run_num=run_num,
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
                                                          objectives_lst=exp_config.optimisation_args.objectives,
                                                          observation=observation,
                                                          physical_pipeline_uuid=physical_pipeline_uuid,
                                                          logical_pipeline_uuid=logical_pipeline_uuid,
                                                          db_client=db_client,
                                                          run_num=run_num)
                # Complete the task
                await task_queue.complete_task(exp_config_name=exp_config_name, run_num=run_num, task_uuid=task_uuid)
                logger.info(f'Task with task_uuid = {task_uuid} and run_num = {run_num} was successfully completed.')

            except Exception as err:
                logger.error(f'Consume error: {err}')
                await consumer.stop()
                consumer = get_kafka_consumer()
                await consumer.start()
    finally:
        await consumer.stop()
