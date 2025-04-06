import asyncio
from munch import DefaultMunch
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from .database.task_manager_db_client import TaskManagerDBClient
from .domain_logic.kafka_processes import start_task_provider, start_cost_model_updater
from .domain_logic.initial_configuration import start_task_generator, create_init_state_for_config
from virny_flow.configs.structs import BOAdvisorConfig
from ..core.custom_classes.async_counter import AsyncCounter

cors = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, HEAD, OPTIONS',
}

def register_routes(app: FastAPI, exp_config: DefaultMunch, db_client: TaskManagerDBClient,
                    uvicorn_server, lp_to_advisor: dict, bo_advisor_config: BOAdvisorConfig,
                    total_pipelines_counter: AsyncCounter):
    @app.options("/{full_path:path}")
    async def options():
        return JSONResponse(status_code=status.HTTP_200_OK, headers=cors, content=None)

    @app.on_event("startup")
    async def startup_event():
        print("Starting up...", flush=True)

        # Create an optimized execution plan
        db_client.connect()
        for run_num in exp_config.common_args.run_nums:
            await create_init_state_for_config(exp_config=exp_config, db_client=db_client, run_num=run_num)
        db_client.close()

        # Start a background process that adds new tasks to the queue in the database if it has available space
        asyncio.create_task(start_task_generator(exp_config=exp_config,
                                                 lp_to_advisor=lp_to_advisor,
                                                 bo_advisor_config=bo_advisor_config,
                                                 total_pipelines_counter=total_pipelines_counter))
        # Start a background process that reads new tasks from the task queue in DB and adds to a Kafka queue
        asyncio.create_task(start_task_provider(exp_config=exp_config,
                                                uvicorn_server=uvicorn_server,
                                                total_pipelines_counter=total_pipelines_counter))
        # Start a background process that updates cost models based on completed tasks
        asyncio.create_task(start_cost_model_updater(exp_config=exp_config,
                                                     lp_to_advisor=lp_to_advisor,
                                                     total_pipelines_counter=total_pipelines_counter))

    @app.on_event("shutdown")
    async def shutdown_event():
        print("Shutting down...")
