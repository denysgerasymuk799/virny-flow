import asyncio
from fastapi import FastAPI, status, Query, Body
from fastapi.responses import JSONResponse
from munch import DefaultMunch
from openbox.utils.history import Observation

from .database.task_manager_db_client import TaskManagerDBClient
from .domain_logic.initial_configuration import add_new_tasks, create_init_state_for_config
from .domain_logic.bayesian_optimization import parse_config_space
from virny_flow.configs.constants import LOGICAL_PIPELINE_SCORES_TABLE
from virny_flow.configs.structs import BOAdvisorConfig
from virny_flow.core.custom_classes.task_queue import TaskQueue


cors = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, HEAD, OPTIONS',
}

def register_routes(app: FastAPI, exp_config: DefaultMunch, task_queue: TaskQueue, db_client: TaskManagerDBClient,
                    lp_to_advisor: dict, bo_advisor_config: BOAdvisorConfig, logger):
    @app.options("/{full_path:path}")
    async def options():
        return JSONResponse(status_code=status.HTTP_200_OK, headers=cors, content=None)

    @app.on_event("startup")
    async def startup_event():
        print("Starting up...")
        db_client.connect()
        task_queue.connect()

        # Create an optimized execution plan
        await create_init_state_for_config(exp_config=exp_config, db_client=db_client)

        # Start a background process that adds new tasks to the queue if it has available space
        asyncio.create_task(add_new_tasks(exp_config=exp_config,
                                          lp_to_advisor=lp_to_advisor,
                                          bo_advisor_config=bo_advisor_config,
                                          db_client=db_client,
                                          task_queue=task_queue))

    @app.on_event("shutdown")
    def shutdown_event():
        print("Shutting down...")
        db_client.close()
        task_queue.close()

    @app.get("/get_worker_task", response_class=JSONResponse)
    async def get_worker_task(exp_config_name: str = Query()):
        high_priority_task = await task_queue.dequeue(exp_config_name=exp_config_name)
        logger.info(f'New task was retrieved, UUID: {high_priority_task["task_uuid"]}')
        return JSONResponse(content=high_priority_task,
                            status_code=status.HTTP_200_OK)

    @app.post("/complete_worker_task", response_class=JSONResponse)
    async def complete_worker_task(data: dict = Body()):
        # Process body
        exp_config_name = data["exp_config_name"]
        task_uuid = data["task_uuid"]
        logical_pipeline_uuid = data["logical_pipeline_uuid"]
        logical_pipeline_name = data["logical_pipeline_name"]

        data["observation"]["config"] = parse_config_space(data["observation"]["config"])
        observation = Observation.from_dict(data["observation"],
                                            config_space=lp_to_advisor[logical_pipeline_name]["config_space"])

        print("exp_config_name:", exp_config_name)
        print("task_uuid:", task_uuid)

        # Update the number of trials for the logical pipeline
        await db_client.increment_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                        condition={"exp_config_name": exp_config_name,
                                                   "logical_pipeline_uuid": logical_pipeline_uuid},
                                        increment_val_dct={"num_trials": 1})

        # Update the advisor of the logical pipeline
        lp_to_advisor[logical_pipeline_name]["config_advisor"].update_observation(observation)

        # Complete the task
        done_tasks_count = await task_queue.complete_task(exp_config_name=exp_config_name,
                                                          task_uuid=task_uuid)
        logger.info(f'Task with task_uuid = {task_uuid} was successfully completed.')
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"message": f"Marked {done_tasks_count} document(s) as DONE"})
