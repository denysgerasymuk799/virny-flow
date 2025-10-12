import os
import time
import asyncio
import json
from munch import DefaultMunch
from datetime import datetime, timezone
from openbox.utils.history import Observation
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .score_model import update_logical_pipeline_score_model
from .bayesian_optimization import parse_config_space
from ..database.task_manager_db_client import TaskManagerDBClient
from ...core.custom_classes.async_counter import AsyncCounter
from ...core.utils.optimization_utils import has_remaining_time_budget, has_remaining_pipelines_budget
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.core.custom_classes.task_queue import TaskQueue
from virny_flow.configs.constants import (TASK_MANAGER_CONSUMER_GROUP, NEW_TASKS_QUEUE_TOPIC, LOGICAL_PIPELINE_SCORES_TABLE, NO_TASKS,
                                          EXP_CONFIG_HISTORY_TABLE, COMPLETED_TASKS_QUEUE_TOPIC, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE)


async def start_task_provider(exp_config: DefaultMunch, uvicorn_server, total_pipelines_counter: AsyncCounter,
                              kafka_broker_address: str):
    termination_flag = False
    empty_execution_budget_flag = False
    logger = get_logger('TaskProvider')
    db_client = TaskManagerDBClient(exp_config.common_args.secrets_path)
    task_queue = TaskQueue(secrets_path=exp_config.common_args.secrets_path,
                           max_queue_size=exp_config.optimisation_args.queue_size)
    db_client.connect()
    task_queue.connect()

    producer = AIOKafkaProducer(bootstrap_servers=kafka_broker_address)
    await producer.start()

    start_time = time.perf_counter()
    for idx, run_num in enumerate(exp_config.common_args.run_nums):
        execution_start_time = time.time()
        print('#' * 40 + '\n' + f'START TASK PROVIDING FOR RUN_NUM={run_num}' + '\n' + '#' * 40, flush=True)

        while True:
            try:
                # Check execution budget conditions
                if (exp_config.optimisation_args.max_time_budget is not None
                    and not has_remaining_time_budget(exp_config.optimisation_args.max_time_budget, start_time)) or \
                        (exp_config.optimisation_args.max_total_pipelines_num is not None
                         and not await has_remaining_pipelines_budget(exp_config.optimisation_args.max_total_pipelines_num, total_pipelines_counter)):
                    # Save execution time of all pipelines for the define experimental config
                    exp_config_execution_time = time.time() - execution_start_time
                    await db_client.update_query(collection_name=EXP_CONFIG_HISTORY_TABLE,
                                                 exp_config_name=exp_config.common_args.exp_config_name,
                                                 run_num=run_num,
                                                 condition={},
                                                 update_val_dct={"exp_config_execution_time": exp_config_execution_time})

                    # If all work is done, set num_available_tasks to num_workers and get new tasks from task_queue.
                    # When task_queue is empty, it will return NO_TASKS, and it will be sent to each worker.
                    num_available_tasks = exp_config.optimisation_args.num_workers * 2  # Multiply by 2 to be sure to shutdown all the workers
                    termination_flag = True
                    empty_execution_budget_flag = True
                    logger.info("An execution budget has expired. Shutting down...")

                else:
                    num_available_tasks = await task_queue.get_num_available_tasks(exp_config_name=exp_config.common_args.exp_config_name, run_num=run_num)
                    if num_available_tasks == 0:
                        if not await task_queue.is_empty(exp_config_name=exp_config.common_args.exp_config_name, run_num=run_num):
                            print("Wait until new tasks come up...", flush=True)
                            await asyncio.sleep(10)
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
                            await asyncio.sleep(10)
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
                        producer = AIOKafkaProducer(bootstrap_servers=kafka_broker_address)
                        await producer.start()

                if termination_flag:
                    # Terminate CostModelUpdater
                    terminate_consumer_msg = {"exp_config_name": exp_config.common_args.exp_config_name, "run_num": run_num, "task_uuid": NO_TASKS}
                    terminate_consumer_json_msg = json.dumps(terminate_consumer_msg)
                    await producer.send_and_wait(topic=COMPLETED_TASKS_QUEUE_TOPIC,
                                                 value=terminate_consumer_json_msg.encode('utf-8'))
                    # Terminate TaskProvider
                    db_client.close()
                    task_queue.close()

                    try:
                        # Terminate TaskManager web-server
                        uvicorn_server.should_exit = True
                    except Exception as e:
                        logger.error(f'Error during uvicorn server termination: {e}')

                    if empty_execution_budget_flag:
                        await producer.stop()
                        return
                    else:
                        break

            except Exception as err:
                logger.error(f'Produce error: {err}')
                await producer.stop()
                producer = AIOKafkaProducer(bootstrap_servers=kafka_broker_address)
                await producer.start()

    await producer.stop()


def get_kafka_consumer(kafka_broker_address):
    return AIOKafkaConsumer(COMPLETED_TASKS_QUEUE_TOPIC,
                            bootstrap_servers=[kafka_broker_address],
                            group_id=TASK_MANAGER_CONSUMER_GROUP,
                            session_timeout_ms=300_000,  # Increase session timeout (default: 10000 ms)
                            heartbeat_interval_ms=20_000,  # Increase heartbeat interval (default: 3000 ms)
                            max_poll_interval_ms=600_000,  # Increase to 10 minutes if needed
                            max_poll_records=1,
                            request_timeout_ms=330_000,
                            auto_offset_reset="earliest",
                            enable_auto_commit=True)


async def start_cost_model_updater(exp_config: DefaultMunch, lp_to_advisor: dict, total_pipelines_counter: AsyncCounter,
                                   kafka_broker_address: str):
    logger = get_logger('CostModelUpdater')
    db_client = TaskManagerDBClient(exp_config.common_args.secrets_path)
    task_queue = TaskQueue(secrets_path=exp_config.common_args.secrets_path,
                           max_queue_size=exp_config.optimisation_args.queue_size)
    db_client.connect()
    task_queue.connect()

    consumer = get_kafka_consumer(kafka_broker_address)
    await consumer.start()

    start_time = time.perf_counter()
    try:
        async for record in consumer:
            try:
                data = json.loads(record.value)
                logger.info(f'Consumed a new task for cost model updating: {data["task_uuid"]}')

                # Process body
                exp_config_name = data["exp_config_name"]
                run_num = data["run_num"]
                task_uuid = data["task_uuid"]

                # Check termination state
                if task_uuid == NO_TASKS:
                    logger.info('Shutting down Cost Model Updater...')
                    db_client.close()
                    task_queue.close()
                    break

                physical_pipeline_uuid = data["physical_pipeline_uuid"]
                logical_pipeline_uuid = data["logical_pipeline_uuid"]
                logical_pipeline_name = data["logical_pipeline_name"]

                data["observation"]["config"] = parse_config_space(data["observation"]["config"])
                observation = Observation.from_dict(data["observation"],
                                                    config_space=lp_to_advisor[run_num][logical_pipeline_name]["config_space"])

                # Update the advisor of the logical pipeline
                lp_to_advisor[run_num][logical_pipeline_name]["config_advisor"].update_observation(observation)

                # Update score of the selected logical pipeline
                compound_pp_quality = await update_logical_pipeline_score_model(exp_config_name=exp_config_name,
                                                                                objectives_lst=exp_config.optimisation_args.objectives,
                                                                                observation=observation,
                                                                                logical_pipeline_uuid=logical_pipeline_uuid,
                                                                                db_client=db_client,
                                                                                run_num=run_num)
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
                                                          "compound_pp_quality": compound_pp_quality,
                                                          "create_datetime": datetime_now,
                                                          "update_datetime": datetime_now,
                                                      })
                # Complete the task
                await task_queue.complete_task(exp_config_name=exp_config_name, run_num=run_num, task_uuid=task_uuid)
                logger.info(f'Task with task_uuid = {task_uuid} and run_num = {run_num} was successfully completed.')

                # Increment the total number of successfully executed pipelines
                await total_pipelines_counter.increment()
                total_pipelines_num = await total_pipelines_counter.get_value()
                logger.info(f'The number of successfully executed pipelines is {total_pipelines_num}.')
                logger.info(f'System execution runtime is {time.perf_counter() - start_time}.')

                # Check execution budget conditions
                if (exp_config.optimisation_args.max_time_budget is not None
                    and not has_remaining_time_budget(exp_config.optimisation_args.max_time_budget, start_time)) or \
                        (exp_config.optimisation_args.max_total_pipelines_num is not None
                         and not await has_remaining_pipelines_budget(exp_config.optimisation_args.max_total_pipelines_num, total_pipelines_counter)):
                    db_client.close()
                    task_queue.close()
                    logger.info("An execution budget has expired. Shutting down...")
                    return

            except Exception as err:
                logger.error(f'Consume error: {err}')
                await consumer.stop()
                consumer = get_kafka_consumer(kafka_broker_address)
                await consumer.start()
    finally:
        await consumer.stop()
