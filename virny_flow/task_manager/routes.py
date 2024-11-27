import asyncio
from munch import DefaultMunch
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from .database.task_manager_db_client import TaskManagerDBClient
from .domain_logic.kafka_processes import start_task_provider, start_cost_model_updater
from .domain_logic.initial_configuration import start_task_generator, create_init_state_for_config
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

        # Start a background process that adds new tasks to the queue in the database if it has available space
        asyncio.create_task(start_task_generator(exp_config=exp_config,
                                                 lp_to_advisor=lp_to_advisor,
                                                 bo_advisor_config=bo_advisor_config,
                                                 db_client=db_client,
                                                 task_queue=task_queue))
        # Start a background process that reads new tasks from the task queue in DB and adds to a Kafka queue
        asyncio.create_task(start_task_provider(exp_config=exp_config,
                                                db_client=db_client,
                                                task_queue=task_queue))
        # Start a background process that updates cost models based on completed tasks
        asyncio.create_task(start_cost_model_updater(exp_config=exp_config,
                                                     lp_to_advisor=lp_to_advisor,
                                                     db_client=db_client,
                                                     task_queue=task_queue))

    @app.on_event("shutdown")
    def shutdown_event():
        print("Shutting down...")
        db_client.close()
        task_queue.close()
